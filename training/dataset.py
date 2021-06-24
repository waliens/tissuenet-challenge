import os
from abc import abstractmethod
from collections import defaultdict

import PIL
import cv2
import math
import numpy as np
import sldc
import torch
from PIL import Image
from cytomine.models import Annotation
from rasterio.features import rasterize
from shapely import wkt
from shapely.affinity import translate, affine_transform
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from sldc import TileTopology
from sldc.image import FixedSizeTileTopology, DefaultTileBuilder
from sldc_cytomine import CytomineTileBuilder, CytomineSlide
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class PilImage(sldc.Image):
    def __init__(self, filepath):
        self._filepath = filepath
        self._image = cv2.imread(self._filepath)[:, :, ::-1]

    @property
    def image(self):
        return self._image

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def channels(self):
        return self.image.shape[-1]

    @property
    def np_image(self):
        if self.image.ndim == 0:
            raise ValueError("image empty '{}'".format(self._filepath))
        return self.image


def powdiv(v, p):
    return v / (2 ** p)


def convert_poly(p, zoom, im_height):
    """Move a polygon to the correct zoom level and referential"""
    polygon = affine_transform(p, [powdiv(1, zoom), 0, 0, powdiv(1, zoom), 0, 0])
    return affine_transform(polygon, [1, 0, 0, -1, 0, im_height])


class BaseAnnotationCrop(object):
    @abstractmethod
    def random_crop_and_mask(self):
        pass

    @abstractmethod
    def crop_and_mask(self):
        pass


class AnnotationCrop(BaseAnnotationCrop):
    def __init__(self, wsi, annotation, working_path, tile_size=512, zoom_level=0, n_jobs=0, intersecting=None):
        self._annotation = annotation
        self._tile_size = tile_size
        self._wsi = CytomineSlide(wsi, zoom_level=zoom_level)
        self._builder = CytomineTileBuilder(working_path, n_jobs=n_jobs)
        self._working_path = working_path
        self._zoom_level = zoom_level
        self._other_annotations = [] if intersecting is None else intersecting
        self._other_polygons = [self._annot2poly(a) for a in self._other_annotations]

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def wsi(self):
        return self._wsi

    @property
    def image_instance(self):
        return self._wsi.image_instance

    @property
    def annotation(self):
        return self._annotation

    @property
    def polygon(self):
        return self._polygon()

    @property
    def image_box(self):
        return self._extract_image_box()

    def _get_start_and_size_over_dimension(self, crop_start, crop_size, wsi_size):
        start = crop_start
        size = crop_size
        if crop_size < self._tile_size:
            start = crop_start + (crop_size - self._tile_size) // 2
            size = self._tile_size
        # make sure that the tile is in the image
        start = max(0, start)
        start = min(start, wsi_size - size)
        if start < 0:
            raise ValueError("image is smaller than the tile size")
        return start, size

    def _extract_image_box(self):
        crop_width, crop_height = self._crop_dims()
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self._crop_bounds()
        image_x_min, image_width = self._get_start_and_size_over_dimension(crop_x_min, crop_width, self._wsi.width)
        image_y_min, image_height = self._get_start_and_size_over_dimension(crop_y_min, crop_height, self._wsi.height)
        return (image_x_min, image_y_min), image_width, image_height

    def _get_image_filepath(self):
        (x, y), width, height = self._extract_image_box()
        return os.path.join(self._working_path, "{}-{}-{}-{}-{}-{}.png").format(self._zoom_level, self.image_instance.id, x, y, width, height)

    def _download_image(self):
        filepath = self._get_image_filepath()
        if not os.path.isfile(filepath):
            (x, y), width, height = self._extract_image_box()
            tile = self._wsi.tile(self._builder, (x, y), width, height)
            image = PIL.Image.fromarray(tile.np_image)
            image.save(filepath)
        return filepath

    def download(self, verbose=False):
        if verbose:
            print("download '{}'".format(self._get_image_filepath()))
        return self._download_image()

    def _polygon(self):
        return self._annot2poly(self._annotation)

    def _annot2poly(self, annot):
        polygon = wkt.loads(annot.location)
        return convert_poly(polygon, self._zoom_level, self.wsi.height)

    def _crop_bounds(self):
        """at the specified zoom level"""
        x_min, y_min, x_max, y_max = self._polygon().bounds
        return int(x_min), int(y_min), math.ceil(x_max), math.ceil(y_max)

    def _crop_dims(self):
        x_min, y_min, x_max, y_max = self._crop_bounds()
        return x_max - x_min, y_max - y_min

    def _robust_load_crop(self, x, y):
        attempts = 0
        filepath = self._get_image_filepath()
        while True:
            try:
                return Image.open(filepath).crop([x, y, x + self._tile_size, y + self._tile_size])
            except OSError as e:
                if attempts > 3:
                    raise e
                print("recreate '{}'".format(filepath))
                os.remove(filepath)
                self.download()

    def _robust_load_image(self):
        attempts = 0
        filepath = self._get_image_filepath()
        while True:
            try:
                return Image.open(filepath)
            except OSError as e:
                if attempts > 3:
                    raise e
                print("recreate '{}'".format(filepath))
                os.remove(filepath)
                self.download()

    def random_crop_and_mask(self):
        """in image coordinate system"""
        (x_min, y_min), width, height = self._extract_image_box()
        x = np.random.randint(0, width - self._tile_size + 1)
        y = np.random.randint(0, height - self._tile_size + 1)
        crop = self._robust_load_crop(x, y)
        mask = self._mask(x, y, self._tile_size, self._tile_size)
        return (x, y, self._tile_size, self._tile_size), crop, Image.fromarray(mask.astype(np.uint8))

    def crop_and_mask(self):
        """in image coordinates system, get full crop and mask"""
        _, width, height = self._extract_image_box()
        image = self._robust_load_image()
        mask = self._mask(0, 0, width, height)
        return image, Image.fromarray(mask.astype(np.uint8))

    def _mask(self, window_x, window_y, window_width, window_height):
        (crop_x, crop_y), crop_width, crop_height = self.image_box
        ground_truth = [self._polygon()] + self._other_polygons
        window = box(0, 0, window_width, window_height)
        fg = [translate(g, xoff=-(window_x + crop_x), yoff=-(window_y + crop_y)).intersection(window)
              for g in ground_truth]
        fg = [p for p in fg if not p.is_empty]
        if len(fg) > 0:
            mask = rasterize(fg, out_shape=(window_height, window_width), fill=0, dtype=np.uint8) * 255
        else:
            mask = np.zeros([window_height, window_width])
        return mask

    @property
    def intersecting(self):
        return self._other_annotations

    @property
    def sldc_image(self):
        return PilImage(self._get_image_filepath())

    @property
    def sldc_window(self):
        xmin, ymin, _, _ = self._crop_bounds()
        width, height = self._crop_dims()
        return self._wsi.window((xmin, ymin), width, height)

    def topology(self, width, height, overlap=0):
        base_topology = TileTopology(self.sldc_image, tile_builder=self.tile_builder, max_width=width, max_height=height, overlap=overlap)
        return FixedSizeTileTopology(base_topology)

    @property
    def tile_builder(self):
        return DefaultTileBuilder()


class AnnotationCropWithCue(BaseAnnotationCrop):
    def __init__(self, crop: BaseAnnotationCrop, cue, cue_only=False):
        """
        Parameters
        ----------
        crop: BaseAnnotationCrop
        cue: ndarray
            Probability map for the cue np.array of float in [0, 1]
        """
        self._crop = crop
        self._cue = (cue * 255)
        self._cue_only = cue_only

    @property
    def cue_only(self):
        return self._cue_only

    @cue_only.setter
    def cue_only(self, value):
        self._cue_only = value

    def random_crop_and_mask(self):
        crop_location, crop, mask = self._crop.random_crop_and_mask()
        x, y, w, h = crop_location
        final_mask = self._cue[y:(y+h), x:(x+w)]
        if not self.cue_only:
            final_mask[np.asarray(mask) > 0] = 255
        return crop_location, crop, Image.fromarray(final_mask.astype(np.uint8), "L")

    def crop_and_mask(self):
        crop, mask = self._crop.crop_and_mask()
        final_mask = self._cue
        final_mask[np.asarray(mask) > 0] = 255
        return crop, Image.fromarray(final_mask)

    @property
    def cue(self):
        return self._cue

    @property
    def crop(self):
        return self._crop


class RemoteAnnotationCropTrainDataset(Dataset):
    def __init__(self, crops, image_trans=None, both_trans=None, mask_trans=None):
        self._crops = crops
        self._both_trans = both_trans
        self._image_trans = image_trans
        self._mask_trans = mask_trans

    def __getitem__(self, item):
        annotation_crop = self._crops[item]
        _, image, mask = annotation_crop.random_crop_and_mask()

        if self._both_trans is not None:
            image, mask = self._both_trans([image, mask])
        if self._image_trans is not None:
            image = self._image_trans(image)
        if self._mask_trans is not None:
            mask = self._mask_trans(mask)

        return image, mask

    def __len__(self):
        return len(self._crops)


class TileTopologyDataset(Dataset):
    def __init__(self, topology, trans=None):
        self._topology = topology
        self._trans = trans

    @property
    def topology(self):
        return self._topology

    @property
    def trans(self):
        return self._trans

    def __getitem__(self, item):
        image = Image.fromarray(self._topology.tile(item + 1).np_image)
        if self._trans is not None:
            image = self._trans(image)
        return item + 1, image

    def __len__(self):
        return len(self._topology)


def predict_roi(roi, ground_truth, model, device, in_trans=None, batch_size=1, tile_size=256, overlap=0, n_jobs=1, zoom_level=0):
    """
    Parameters
    ----------
    roi: AnnotationCrop
        The polygon representing the roi to process
    ground_truth: iterable of Annotation|Polygon
        The ground truth annotations
    model: nn.Module
        Segmentation network. Takes a batch of _images as input and outputs the foreground probability for all pixels
    device:
        A torch device to transfer data to
    in_trans: transforms.Transform
        A transform to apply before forwarding _images into the network
    batch_size: int
        Batch size
    tile_size: int
        Tile size
    overlap: int
        Tile tile_overlap
    n_jobs: int
        Number of jobs available
    zoom_level: int
        Zoom level

    Returns
    -------
    """
    # topology
    tile_topology = roi.topology(width=tile_size, height=tile_size, overlap=overlap)
    (x_min, y_min), width, height = roi.image_box
    mask_dims = (height, width)

    # build ground truth
    roi_poly = roi.polygon
    ground_truth = [(wkt.loads(g.location) if isinstance(g, Annotation) else g) for g in ground_truth]
    ground_truth = [convert_poly(g, zoom_level, roi.wsi.height) for g in ground_truth]
    translated_gt = [translate(g.intersection(roi_poly), xoff=-x_min, yoff=-y_min) for g in ground_truth]

    y_true = rasterize(translated_gt, out_shape=mask_dims, fill=0, dtype=np.uint8)
    y_pred = np.zeros(y_true.shape, dtype=np.double)
    y_acc = np.zeros(y_true.shape, dtype=np.int)

    # dataset and loader
    dataset = TileTopologyDataset(tile_topology, trans=in_trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_jobs)

    for ids, x in dataloader:
        x = x.to(device)
        y = model.forward(x, sigmoid=True)

        # accumulate predictions
        for i, identifier in enumerate(ids):
            x_off, y_off = tile_topology.tile_offset(identifier.item())
            y_pred[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += y[i].detach().cpu().squeeze().numpy()
            y_acc[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += 1

    # average multiple predictions
    y_pred /= y_acc

    # import cv2
    # from datetime import datetime
    # roi.annotation.dump("{}_image.png".format(roi.annotation.id), override=False)
    # cv2.imwrite("{}_true.png".format(roi.annotation.id), y_true * 255)
    # cv2.imwrite("{}_pred_{}.png".format(roi.annotation.id, datetime.now().timestamp()), (y_pred * 255).astype(np.uint8))
    return y_pred, y_true


def datasets_size_cumsum(datasets):
    sizes = np.array([len(d) for d in datasets])
    cumsum = np.concatenate([np.array([0]), np.cumsum(sizes[:-1], dtype=np.int)])
    return sizes, cumsum


def get_sample_indexes(index, cumsum):
    dataset_index = np.searchsorted(cumsum, index, side="right") - 1
    relative_index = index - cumsum[dataset_index]
    return dataset_index, relative_index


class AnnotationCropTopoplogyDataset(Dataset):
    def __init__(self, crop, overlap=0, in_trans=None):
        self._dataset = TileTopologyDataset(crop.topology(crop.tile_size, crop.tile_size, overlap=overlap), trans=in_trans)
        self._crop = crop

    def __getitem__(self, item):
        _id, tile = self._dataset[item]
        x_off, y_off = self._dataset.topology.tile_offset(_id)
        return _id, x_off, y_off, tile

    def __len__(self):
        return len(self._dataset)


class MultiCropsSet(Dataset):
    def __init__(self, crops, in_trans, overlap=0):
        """
        Parameters
        ----------
        do_add_group: bool
            True to append group identifier (optional), default: `False`.
        kwargs: dict
            Parameters to be transferred to the actual `ImageFolder`.
        """
        super().__init__()
        self._datasets = [
            AnnotationCropTopoplogyDataset(crop, overlap=overlap, in_trans=in_trans)
            for crop in crops]
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        dataset = self._datasets[dataset_index]
        return (dataset._crop.annotation.id,) + dataset[relative_index]

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def predict_annotation_crops_with_cues(net, crops, device, in_trans=None, overlap=0, batch_size=8, n_jobs=1):
    if len(crops) == 0:
        return 0
    dataset = MultiCropsSet(crops, in_trans=in_trans, overlap=overlap)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=n_jobs, pin_memory=True, drop_last=False)

    tile_size = crops[0].tile_size
    n_bytes = len(dataset) * tile_size * tile_size * 4
    print("> annot with cues needs approx {} of memory".format(sizeof_fmt(n_bytes)), flush=True)
    all_ys = defaultdict(list)
    net.eval()
    for annot_ids, tile_ids, xs, ys, tiles in loader:
        t = tiles.to(device)
        y = torch.sigmoid(net.forward(t))
        detached = y.detach().cpu().numpy()
        for i, (annot_id, tile_id, x_off, y_off) in enumerate(zip(annot_ids, tile_ids, xs, ys)):
            all_ys[annot_id.item()].append((tile_id.item(), (x_off.item(), y_off.item()), detached[i].squeeze()))

    awcues = list()
    for crop in crops:
        _, w, h = crop.image_box
        cue = np.zeros([h, w], dtype=np.float)
        acc = np.zeros([h, w], dtype=np.int)
        for tile_id, (x_off, y_off), y_pred in all_ys[crop.annotation.id]:
            cue[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += y_pred
            acc[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += 1
        cue /= acc
        awcues.append(AnnotationCropWithCue(crop, cue=cue))
        del(all_ys[crop.annotation.id])

    return awcues