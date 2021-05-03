import os
from functools import partial

import numpy as np
from abc import abstractmethod

import torch
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import box
from shapely.ops import cascaded_union
from torch.nn import ConstantPad2d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.resnet import conv1x1

from assets.inference import foreground_detect, TimingContextManager, convex_white_detect
from assets.mtdp import build_model
from assets.sldc.image import FixedSizeTileTopology
from assets.sldc_pyvips.adapter import PyVipsSlide, PyVipsTileBuilder
from encoded_slide_train import CustomSlideClassifier


def check_tile_intersects(tile, polygons):
    x, y = tile.abs_offset
    b = box(x, y, x + tile.width, y + tile.height)
    return np.any([b.overlaps(p) for p in polygons])


class SlideTileDataset(Dataset):
    def __init__(self, topology, check_fn=None, trans=None):
        self._topology = topology
        self._trans = trans
        self._check_fn = (lambda tile: True) if check_fn is None else check_fn
        self._index2identifier = self._prepare()

    def _prepare(self):
        filtered2full_ids = list()
        for tile in self._topology:
            if self._check_fn(tile):
                filtered2full_ids.append(tile.identifier)
        return filtered2full_ids

    def __getitem__(self, item):
        identifier = self._index2identifier[item]
        tile = self._topology.tile(identifier)
        row, col = self._topology._tile_coord(identifier)
        image = Image.fromarray(tile.np_image)
        if self._trans is not None:
            image = self._trans(image)
        return image, (row, col)

    def __len__(self):
        return len(self._index2identifier)


class SlideEncoder(object):
    def __init__(self, zoom_level=0):
        self._zoom_level = zoom_level

    @property
    def zoom_level(self):
        return self._zoom_level

    @abstractmethod
    def encode(self, slide_path):
        pass


class ModelSlideEncoder(SlideEncoder):
    def __init__(self, model, trans=None, tile_size=512, tile_overlap=0, batch_size=16, zoom_level=0, n_jobs=1, bg_exclude=False, crop_fg=False, device=None):
        super().__init__(zoom_level=zoom_level)
        self._model = model
        self._batch_size = batch_size
        self._tile_size = tile_size
        self._overlap = tile_overlap
        self._trans = trans if trans is not None else (lambda i: i)
        self._bg_exclude = bg_exclude
        self._device = device
        self._n_jobs = n_jobs
        self._fg_crop = crop_fg

    def _get_check_fn(self, filepath, tissues=None):
        if not self._bg_exclude:
            return lambda tile: True
        else:
            if tissues is None:
                tissues = self._fg_detect(filepath)
            return partial(check_tile_intersects, polygons=tissues)

    def _fg_detect(self, filepath):
        tissues = foreground_detect(slide_path=filepath)
        return [self._affine_to_zoom(p) for p in tissues]

    def _affine_to_zoom(self, p):
        zoom_ratio = 2 ** self._zoom_level
        return affine_transform(p, [1 / zoom_ratio, 0, 0, 1 / zoom_ratio, 0, 0])

    def _fg_window(self, filepath):
        tissues = self._fg_detect(filepath)
        if len(tissues) == 0:
            tissues = [self._affine_to_zoom(convex_white_detect(filepath))]
        if len(tissues) == 0:
            return tissues, None
        return tissues, box(*cascaded_union(tissues).bounds)

    def encode(self, slide_path):
        slide = PyVipsSlide(slide_path, zoom_level=self.zoom_level)
        tissues = None
        if self._fg_crop:
            tissues, crop_window = self._fg_window(slide_path)
            if crop_window is not None:
                slide = slide.window_from_polygon(crop_window)
            else:
                print("no foreground, cannot crop for image '{}'".format(slide_path))
        builder = PyVipsTileBuilder(slide)
        base_topology = slide.tile_topology(builder, max_width=self._tile_size, max_height=self._tile_size,
                                            overlap=self._overlap)
        topology = FixedSizeTileTopology(base_topology)
        dataset = SlideTileDataset(
            topology,
            check_fn=self._get_check_fn(slide_path, tissues=tissues),
            trans=self._trans)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._n_jobs, pin_memory=True)

        encoded = np.zeros([
            self._model.n_features(),
            topology.tile_vertical_count,
            topology.tile_horizontal_count
        ], dtype=np.float)

        for j, (x, coords) in enumerate(loader):
            x = x.to(self._device)
            y = self._model.forward(x).detach().cpu().numpy().squeeze()

            for i, (row, col) in enumerate(zip(coords[0], coords[1])):
                encoded[:, row.item(), col.item()] = y[i]

        return encoded


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def main():
    PATH = "/scratch/users/rmormont/tissuenet"
    SLIDE_PATH = os.path.join(PATH, "wsis")
    SAVE_PATH = os.path.join(PATH, "wsi_encoded_aug")
    MODEL_PATH = os.path.join(PATH, "models", "continued", "contd_densenet121_mtdp_e_88_z2_1603752206.965628.pth")
    ARCH = "densenet121"
    N_JOBS = 8
    DEVICE = "cuda:0"

    os.makedirs(SAVE_PATH, exist_ok=True)
    device = torch.device(DEVICE)
    features = build_model(arch=ARCH, pretrained=False, pool=True)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    features.load_state_dict({n.split(".", 1)[1]: v for n, v in state_dict.items() if not "linear." in n})
    features.eval()
    features.to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    with torch.no_grad():
        classifier = CustomSlideClassifier()
        classifier.eval()
        classifier.to(device)
        encoder = ModelSlideEncoder(features, trans=trans, tile_size=320, zoom_level=2, n_jobs=N_JOBS, bg_exclude=True, crop_fg=True, device=device)
        timing = TimingContextManager()
        total_bytes = 0
        for filename in os.listdir(SLIDE_PATH):
            filepath = os.path.join(SLIDE_PATH, filename)
            print(filename)
            with timing:
                encoded = encoder.encode(filepath)
            print(timing.duration, "secondes", encoded.shape)
            print("bytes", sizeof_fmt(encoded.nbytes))
            total_bytes += encoded.nbytes
            np.save(os.path.join(SAVE_PATH, "{}.npy".format(filename.rsplit(".", 1)[0])), encoded)

    print("final bytes", sizeof_fmt(total_bytes))


if __name__ == "__main__":
    main()
