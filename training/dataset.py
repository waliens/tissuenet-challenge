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


class BaseCrop(object):
    @abstractmethod
    def random_crop_and_mask(self):
        """
        if has_cue is False, gt_mask and cue_mask are equal to mask.
        crop_loc, crop, gt_mask, cue_mask, mask, has_cue
        """
        pass

    @abstractmethod
    def crop_and_mask(self):
        """
        if has_cue is False, gt_mask and cue_mask are equal to mask.
        crop, gt_mask, cue_mask, mask, has_cue
        """
        pass

    @abstractmethod
    def topology(self, width, height, overlap=0):
        return

    @property
    @abstractmethod
    def width(self):
        pass

    @property
    @abstractmethod
    def height(self):
        pass

    @property
    @abstractmethod
    def height_reference(self):
        pass

    @property
    @abstractmethod
    def offset(self):
        pass

    @property
    @abstractmethod
    def tile_size(self):
        pass

    @property
    @abstractmethod
    def unique_identifier(self):
        pass


class AnnotationCrop(BaseCrop):
    def __init__(self, wsi, annotation, working_path, tile_size=512, zoom_level=0, n_jobs=0, intersecting=None):
        self._annotation = annotation
        self._tile_size = tile_size
        self._wsi = CytomineSlide(wsi, zoom_level=zoom_level)
        self._builder = CytomineTileBuilder(working_path, n_jobs=n_jobs)
        self._working_path = working_path
        self._zoom_level = zoom_level
        self._other_annotations = [] if intersecting is None else intersecting
        self._other_polygons = [self._annot2poly(a) for a in self._other_annotations]

        # load in memory
        _, width, height = self._extract_image_box()
        self._cache_mask = Image.fromarray(self._make_mask(0, 0, width, height).astype(np.uint8))

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

    @property
    def width(self):
        return self._extract_image_box()[1]

    @property
    def height(self):
        return self._extract_image_box()[2]

    @property
    def height_reference(self):
        return self.image_instance.height

    @property
    def offset(self):
        return self._extract_image_box()[0]

    @property
    def unique_identifier(self):
        return self._annotation.id

    @staticmethod
    def get_start_size_ove_dimension(crop_start, crop_size, wsi_size, tile_size):
        start = crop_start
        size = crop_size
        if crop_size < tile_size:
            start = crop_start + (crop_size - tile_size) // 2
            size = tile_size
        # make sure that the tile is in the image
        start = max(0, start)
        start = min(start, wsi_size - size)
        if start < 0:
            raise ValueError("image is smaller than the tile size")
        return start, size

    def _get_start_and_size_over_dimension(self, crop_start, crop_size, wsi_size):
        return AnnotationCrop.get_start_size_ove_dimension(crop_start, crop_size, wsi_size, self._tile_size)

    def _extract_image_box(self):
        crop_width, crop_height = self._crop_dims()
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self._crop_bounds()
        image_x_min, image_width = self._get_start_and_size_over_dimension(crop_x_min, crop_width, self._wsi.width)
        image_y_min, image_height = self._get_start_and_size_over_dimension(crop_y_min, crop_height, self._wsi.height)
        return (image_x_min, image_y_min), image_width, image_height

    def _get_image_filepath(self):
        (x, y), width, height = self._extract_image_box()
        return os.path.join(self._working_path, "{}-{}-{}-{}-{}-{}.png").format(
            self._zoom_level, self.image_instance.id, x, y, width, height)

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

    def _crop_from_cache(self, x, y):
        crop_array = [x, y, x + self._tile_size, y + self._tile_size]
        return self._robust_load_image().crop(crop_array), self._cache_mask.crop(crop_array)

    def _robust_load_image(self):
        attempts = 0
        filepath = self._get_image_filepath()
        while True:
            try:
                image = Image.open(filepath)
                image.load()
                return image
            except OSError as e:
                if attempts > 3:
                    raise e
                print("recreate '{}'".format(filepath))
                if os.path.exists(filepath):
                    os.remove(filepath)
                self.download()

    def random_crop_and_mask(self):
        """in image coordinate system"""
        (_, _), width, height = self._extract_image_box()
        x = np.random.randint(0, width - self._tile_size + 1)
        y = np.random.randint(0, height - self._tile_size + 1)
        crop, mask = self._crop_from_cache(x, y)
        return (x, y, self._tile_size, self._tile_size), crop, mask, mask, mask, False

    def crop_and_mask(self):
        """in image coordinates system, get full crop and mask"""
        _, width, height = self._extract_image_box()
        image, mask = self._robust_load_image(), self._cache_mask
        return image, mask, mask, mask, False

    def _make_mask(self, window_x, window_y, window_width, window_height):
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
        base_topology = TileTopology(self.sldc_image, tile_builder=self.tile_builder, max_width=width,
                                     max_height=height, overlap=overlap)
        return FixedSizeTileTopology(base_topology)

    @property
    def tile_builder(self):
        return DefaultTileBuilder()


class MemoryCrop(BaseCrop):
    def __init__(self, img_path, mask_path, tile_size=256):
        self._img_path = img_path
        self._mask_path = mask_path
        self._image = Image.open(self._img_path)
        self._image.load()
        self._mask = Image.open(self._mask_path)
        self._mask.load()
        self._tile_size = tile_size

    @property
    def unique_identifier(self):
        return os.path.basename(self._img_path)

    @property
    def tile_size(self):
        return self._tile_size

    @property
    def img_path(self):
        return self._img_path

    @property
    def width(self):
        return self._image.width

    @property
    def height(self):
        return self._image.height

    @property
    def height_reference(self):
        return self.height

    @property
    def offset(self):
        return 0, 0

    def crop_and_mask(self):
        return self._image, self._mask, self._mask, self._mask, False

    def random_crop_and_mask(self):
        width, height = self._image.width, self._image.height
        x = np.random.randint(0, width - self._tile_size + 1)
        y = np.random.randint(0, height - self._tile_size + 1)
        img_crop = self._image.crop([x, y, x + self._tile_size, y + self._tile_size])
        mask_crop = self._mask.crop([x, y, x + self._tile_size, y + self._tile_size])
        return (x, y, self._tile_size, self._tile_size), img_crop, mask_crop, mask_crop, mask_crop, False

    def topology(self, width, height, overlap=0):
        base = TileTopology(PilImage(self._img_path), tile_builder=DefaultTileBuilder(),
                            max_width=width, max_height=height, overlap=0)
        return FixedSizeTileTopology(base)


class CropWithCue(BaseCrop):
    def __init__(self, crop: BaseCrop, cue, cue_only=False):
        """
        Parameters
        ----------
        crop: BaseCrop
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
        crop_location, crop, mask, _, _, _ = self._crop.random_crop_and_mask()
        x, y, w, h = crop_location
        final_mask = self._cue[y:(y + h), x:(x + w)]
        if not self.cue_only:
            final_mask[np.asarray(mask) > 0] = 255
        return (crop_location,
                crop,
                mask,
                Image.fromarray(self._cue[y:(y + h), x:(x + w)], "L"),
                Image.fromarray(final_mask.astype(np.uint8), "L"),
                True)

    def crop_and_mask(self):
        crop, mask, _, _, _ = self._crop.crop_and_mask()
        final_mask = np.copy(self._cue)
        final_mask[np.asarray(mask) > 0] = 255
        return crop, mask, Image.fromarray(self._cue, "L"), Image.fromarray(final_mask), True

    @property
    def cue(self):
        return self._cue

    @property
    def crop(self):
        return self._crop


class CropWithThresholdedCue(CropWithCue):
    def __init__(self, base_crop: CropWithCue, threshold=127, cue_only=False):
        super().__init__(base_crop.crop, (base_crop.cue >= threshold).astype(np.uint8), cue_only=cue_only)
        self._base_crop = base_crop
        self._threshold = threshold

    def random_crop_and_mask(self):
        crop_loc, crop, gt_mask, _, mask, has_cue = super().random_crop_and_mask()
        x, y, w, h = crop_loc
        return crop_loc, crop, gt_mask, Image.fromarray(self.soft_cue[y:y+h, x:x+w], 'L'), mask, has_cue

    def crop_and_mask(self):
        crop, gt_mask, _, mask, has_cue = super().crop_and_mask()
        return crop, gt_mask, Image.fromarray(self.soft_cue, 'L'), mask, has_cue

    @property
    def soft_cue(self):
        return self._base_crop.cue



class CropTrainDataset(Dataset):
    def __init__(self, crops, image_trans=None, both_trans=None, mask_trans=None):
        self._crops = crops
        self._both_trans = both_trans
        self._image_trans = image_trans
        self._mask_trans = mask_trans

    def __getitem__(self, item):
        crop = self._crops[item]
        _, image, gt_mask, cue_mask, mask, has_cue = crop.random_crop_and_mask()

        if self._both_trans is not None:
            image, gt_mask, mask, cue_mask = self._both_trans([image, gt_mask, mask, cue_mask])
        if self._image_trans is not None:
            image = self._image_trans(image)
        if self._mask_trans is not None:
            mask = self._mask_trans(mask)
            gt_mask = self._mask_trans(gt_mask)
            cue_mask = self._mask_trans(cue_mask)

        return image, gt_mask, cue_mask, mask, has_cue

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


def predict_roi(roi, ground_truth, model, device, in_trans=None, batch_size=1, tile_size=256, overlap=0, n_jobs=1,
                zoom_level=0):
    """
    Parameters
    ----------
    roi: BaseCrop
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
    (x_min, y_min), width, height = roi.offset, roi.width, roi.height
    mask_dims = (height, width)

    # build ground truth
    if len(ground_truth) > 0:
        roi_poly = box(x_min, y_min, x_min + width, y_min + height)
        ground_truth = [(wkt.loads(g.location) if isinstance(g, Annotation) else g) for g in ground_truth]
        ground_truth = [convert_poly(g, zoom_level, roi.height_reference) for g in ground_truth]
        translated_gt = [translate(g.intersection(roi_poly), xoff=-x_min, yoff=-y_min) for g in ground_truth]

        y_true = rasterize(translated_gt, out_shape=mask_dims, fill=0, dtype=np.uint8)
    else:
        y_true = np.zeros(mask_dims, dtype=np.uint8)
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


class CropTopoplogyDataset(Dataset):
    def __init__(self, crop, overlap=0, in_trans=None):
        self._dataset = TileTopologyDataset(crop.topology(crop.tile_size, crop.tile_size, overlap=overlap),
                                            trans=in_trans)
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
            CropTopoplogyDataset(crop, overlap=overlap, in_trans=in_trans)
            for crop in crops]
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        dataset = self._datasets[dataset_index]
        return (dataset._crop.unique_identifier,) + dataset[relative_index]

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def predict_set(net, crops, device, in_trans, overlap=0, batch_size=8, n_jobs=1, worker_init_fn=None, progress_fn=None):
    if len(crops) == 0:
        return list()
    dataset = MultiCropsSet(crops, in_trans=in_trans, overlap=overlap)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=n_jobs, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    tile_size = crops[0].tile_size
    n_bytes = len(dataset) * tile_size * tile_size * 4
    print("> predicting for current set needs approx {} of memory".format(sizeof_fmt(n_bytes)), flush=True)
    all_ys = defaultdict(list)
    net.eval()
    for annot_ids, tile_ids, xs, ys, tiles in loader:
        t = tiles.to(device)
        y = torch.sigmoid(net.forward(t))
        detached = y.detach().cpu().numpy()
        for i, (annot_id, tile_id, x_off, y_off) in enumerate(zip(annot_ids, tile_ids, xs, ys)):
            all_ys[annot_id].append((tile_id.item(), (x_off.item(), y_off.item()), detached[i].squeeze()))

    all_preds = list()
    for i, crop in enumerate(crops):
        w, h = crop.width, crop.height
        pred = np.zeros([h, w], dtype=np.float)
        acc = np.zeros([h, w], dtype=np.int)
        for tile_id, (x_off, y_off), y_pred in all_ys[crop.unique_identifier]:
            pred[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += y_pred
            acc[y_off:(y_off + tile_size), x_off:(x_off + tile_size)] += 1
        pred /= acc
        del (all_ys[crop.unique_identifier])
        all_preds.append(pred)
        if progress_fn is not None:
            progress_fn(i, len(crops))

    return all_preds


def predict_crops_with_cues(net, crops, device, in_trans=None, overlap=0, batch_size=8, n_jobs=1, progress_fn=None):
    return [
        CropWithCue(crop, cue=pred)
        for crop, pred in zip(crops, predict_set(
                net, crops, device, in_trans=in_trans, overlap=overlap,
                batch_size=batch_size, n_jobs=n_jobs, progress_fn=progress_fn))
    ]


class DatasetsGenerator(object):
    @abstractmethod
    def sets(self):
        """
        Returns
        -------
        incomplete: iterable of BaseCrop
        complete: iterable of BaseCrop
        val_set: iterable of BaseCrop
        """
        pass

    @abstractmethod
    def iterable_to_dataset(self, iterable, **kwargs):
        pass

    @abstractmethod
    def roi_foregrounds(self, val_roi):
        pass

    @abstractmethod
    def crop(self, identifier):
        pass


class GraduallyAddMoreDataState(object):
    def __init__(self, sparse, non_sparse, data_rate=1.0, data_max=1.0):
        self._data_rate = data_rate
        self._data_max = data_max
        self._sparse = sparse
        self._non_sparse = non_sparse
        self._current_amount = 0

    @property
    def abs_data_max(self):
        if self._data_max < 0:
            return min(len(self._non_sparse), len(self._sparse))
        elif 0 <= self._data_max <= 1:
            return int(self._data_max * len(self._sparse))
        else:
            return min(int(self._data_max), len(self._sparse))

    @property
    def abs_date_to_add(self):
        if 0 <= self._data_rate <= 1:
            return int(self._data_rate * len(self._sparse))
        elif self._data_rate > 1:
            return int(self._data_max)

    def get_next(self):
        self._current_amount += self.abs_date_to_add
        self._current_amount = min(self._current_amount, self.abs_data_max)
        return self._sparse[:self._current_amount]
