import os
from functools import partial

import numpy as np
from abc import abstractmethod

import torch
from PIL import Image
from shapely.affinity import affine_transform
from shapely.geometry import box
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from assets.inference import foreground_detect, determine_tissue_extract_level, TimingContextManager
from assets.mtdp import build_model
from assets.sldc.image import FixedSizeTileTopology
from assets.sldc_pyvips.adapter import PyVipsSlide, PyVipsTileBuilder


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
    def __init__(self, model, trans=None, tile_size=512, tile_overlap=0, batch_size=16, zoom_level=0, n_jobs=1, bg_exclude=False, device=None):
        super().__init__(zoom_level=zoom_level)
        self._model = model
        self._batch_size = batch_size
        self._tile_size = tile_size
        self._overlap = tile_overlap
        self._trans = trans if trans is not None else (lambda i: i)
        self._bg_exclude = bg_exclude
        self._device = device
        self._n_jobs = n_jobs

    def _get_check_fn(self, filepath):
        if not self._bg_exclude:
            return lambda tile: True
        else:
            tissues = foreground_detect(slide_path=filepath)
            zoom_ratio = 2 ** self._zoom_level
            tissues = [affine_transform(p, [1 / zoom_ratio, 0, 0, 1 / zoom_ratio, 0, 0]) for p in tissues]
            return partial(check_tile_intersects, polygons=tissues)

    def encode(self, slide_path):
        slide = PyVipsSlide(slide_path, zoom_level=self.zoom_level)
        builder = PyVipsTileBuilder(slide)
        base_topology = slide.tile_topology(builder, max_width=self._tile_size, max_height=self._tile_size,
                                            overlap=self._overlap)
        topology = FixedSizeTileTopology(base_topology)
        dataset = SlideTileDataset(
            topology,
            check_fn=self._get_check_fn(slide_path),
            trans=self._trans)
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._n_jobs, pin_memory=True)

        encoded = np.zeros([
            topology.tile_vertical_count,
            topology.tile_horizontal_count,
            self._model.n_features()
        ], dtype=np.float)

        for j, (x, coords) in enumerate(loader):
            x = x.to(self._device)
            y = self._model.forward(x).detach().cpu().numpy().squeeze()

            for i, (row, col) in enumerate(zip(coords[0], coords[1])):
                encoded[row.item(), col.item()] = y[i]

        return encoded


def main():
    PATH = "/scratch/users/rmormont/tissuenet"
    SLIDE_PATH = os.path.join(PATH, "wsis")
    SAVE_PATH = os.path.join(PATH, "wsi_encoded")
    PRETRAINED = "mtdp"
    ARCH = "densenet121"
    N_JOBS = 8
    DEVICE = "cuda:0"

    device = torch.device(DEVICE)
    features = build_model(arch=ARCH, pretrained=PRETRAINED, pool=True)
    # state_dict = torch.load(MODEL_PATH, map_location=device)
    # features.load_state_dict(state_dict)
    features.eval()
    features.to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    with torch.no_grad():
        encoder = ModelSlideEncoder(features, trans=trans, tile_size=320, zoom_level=0, n_jobs=N_JOBS, bg_exclude=True, device=device)
        timing = TimingContextManager()
        for filename in os.listdir(SLIDE_PATH):
            filepath = os.path.join(SLIDE_PATH, filename)
            print(filename)
            with timing:
                encoded = encoder.encode(filepath)
            print(timing.duration, "secondes", encoded.shape)
            np.save(os.path.join(SAVE_PATH, "{}.npy".format(filename.rsplit(".", 1)[0])), encoded)


if __name__ == "__main__":
    main()
