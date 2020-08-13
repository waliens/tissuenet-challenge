import logging
import os

import cv2
import numpy as np
import sldc
from PIL import Image
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from rasterio.features import rasterize
from shapely import wkt
from shapely.affinity import translate, affine_transform
from shapely.geometry import box
from sldc import TileTopology
from sldc.image import FixedSizeTileTopology, DefaultTileBuilder
from sldc_cytomine import CytomineTileBuilder
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms.functional import vflip, hflip


def segmentation_transform(*images):
    if np.random.rand() > 0.5:
        images = (vflip(i) for i in images)
    if np.random.rand() > 0.5:
        images = (hflip(i) for i in images)
    images = (transforms.ToTensor()(i) for i in images)
    return images


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


class AnnotationCrop(object):
    def __init__(self, wsi, annotation, working_path, tile_size=512):
        self._annotation = annotation
        self._tile_size = tile_size
        self._wsi = wsi
        self._working_path = working_path

    @property
    def wsi(self):
        return self._wsi

    @property
    def annotation(self):
        return self._annotation

    def _get_start_and_size_over_dimension(self, crop_start, crop_size, wsi_size):
        start = crop_start
        size = crop_size
        if crop_size < self._tile_size:
            start = crop_start + (crop_size - self._tile_size) // 2
            size = self._tile_size
        start = min(start, wsi_size - size)
        return start, size

    def _extract_image_box(self):
        crop_width, crop_height = self._crop_dims()
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = self._crop_bounds()
        image_x_min, image_width = self._get_start_and_size_over_dimension(crop_x_min, crop_width, self._wsi.width)
        image_y_min, image_height = self._get_start_and_size_over_dimension(crop_y_min, crop_height, self._wsi.height)
        return (image_x_min, image_y_min), image_width, image_height

    def _get_image_filepath(self):
        (x, y), width, height = self._extract_image_box()
        return os.path.join(self._working_path, "{}-{}-{}-{}-{}.png").format(self._wsi.id, x, y, width, height)

    def _download_image(self):
        (x, y), width, height = self._extract_image_box()
        filepath = self._get_image_filepath()
        if not self._wsi.window(x, y, width, height, dest_pattern=filepath, override=False):
            raise ValueError("could not download the window '{}'".format(filepath))
        self._fix_image()
        return filepath

    def _fix_image(self):
        (x, y), width, height = self._extract_image_box()
        im = Image.open(self._get_image_filepath())
        if im.width == width and im.height == height:
            return
        print("fixing image '{}'".format(self._get_image_filepath()))
        new_width, new_height = width, height
        if im.width != width:
            new_width += 1
        if im.height != height:
            new_height += 1
        new_filepath = os.path.join(self._working_path, "{}-{}-{}-{}-{}.png").format(self._wsi.id, x, y, new_width, new_height)
        if not self._wsi.window(x, y, new_width, new_height, dest_pattern=new_filepath, override=False):
            raise ValueError("could not download the window '{}'".format(new_filepath))
        new_im = Image.open(new_filepath)
        new_im = new_im.crop([0, 0, width, height])
        new_im.save(self._get_image_filepath())
        os.remove(new_filepath)

    def download(self):
        print("download '{}'".format(self._get_image_filepath()))
        return self._download_image()

    def _polygon(self):
        polygon = wkt.loads(self._annotation.location)
        return affine_transform(polygon, [1, 0, 0, -1, 0, self._wsi.height])

    def _crop_bounds(self):
        return [int(v) for v in self._polygon().bounds]

    def _crop_dims(self):
        x_min, y_min, x_max, y_max = self._crop_bounds()
        return x_max - x_min, y_max - y_min

    def random_crop_and_mask(self):
        """in image coordinate system"""
        (x_min, y_min), width, height = self._extract_image_box()
        x = np.random.randint(0, width - self._tile_size + 1)
        y = np.random.randint(0, height - self._tile_size + 1)

        image = Image.open(self._get_image_filepath())
        crop = image.crop([x, y, x + self._tile_size, y + self._tile_size])

        translated = translate(self._polygon(), xoff=-(x_min + x), yoff=-(y_min + y))
        in_window = box(0, 0, self._tile_size, self._tile_size).intersection(translated)
        if in_window.is_empty:
            mask = np.zeros((self._tile_size, self._tile_size), dtype=np.uint8)
        else:
            mask = rasterize([in_window], out_shape=(self._tile_size, self._tile_size), fill=0, dtype=np.uint8) * 255

        return crop, Image.fromarray(mask.astype(np.uint8))

    @property
    def sldc_image(self):
        return PilImage(self._get_image_filepath())

    def topology(self, width, height, overlap=0):
        base_topology = TileTopology(self.sldc_image, tile_builder=self.tile_builder, max_width=width, max_height=height, overlap=overlap)
        return FixedSizeTileTopology(base_topology)

    @property
    def tile_builder(self):
        return DefaultTileBuilder()


class RemoteAnnotationTrainDataset(Dataset):
    def __init__(self, crops, in_trans=None, seg_trans=None):
        self._crops = crops
        self._seg_trans = seg_trans
        self._in_trans = in_trans

    def __getitem__(self, item):
        annotation_crop = self._crops[item]
        image, mask = annotation_crop.random_crop_and_mask()

        if self._seg_trans is not None:
            image, mask = self._seg_trans(image, mask)
        if self._in_trans is not None:
            image = self._in_trans(image)

        return image, mask

    def __len__(self):
        return len(self._crops)


class TileTopologyDataset(Dataset):
    def __init__(self, topology, trans=None):
        self._topology = topology
        self._trans = trans

    def __getitem__(self, item):
        image = Image.fromarray(self._topology.tile(item + 1).np_image)
        if self._trans is not None:
            image = self._trans(image)
        return item + 1, image

    def __len__(self):
        return len(self._topology)


def predict_roi(image, roi, ground_truth, model, device, in_trans=None, batch_size=1, tile_size=256, overlap=0, n_jobs=1):
    """
    Parameters
    ----------
    image: ImageInstance
    roi: CropAnnotation
        The polygon representing the roi to process
    ground_truth: iterable of Annotation
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
        Tile overlap
    n_jobs: int
        Number of jobs available

    Returns
    -------
    """
    # build ground truth
    affine_matrix = [1, 0, 0, -1, 0, image.height]
    roi_poly = affine_transform(wkt.loads(roi.annotation.location), affine_matrix)
    ground_truth = [affine_transform(wkt.loads(g.location), affine_matrix) for g in ground_truth]
    min_x, min_y, max_x, max_y = (int(v) for v in roi_poly.bounds)
    mask_dims = (int(max_x - min_x), int(max_y - min_y))
    translated_gt = [translate(g.intersection(roi_poly), xoff=-min_x, yoff=-min_y) for g in ground_truth]
    y_true = rasterize(translated_gt, out_shape=mask_dims, fill=0, dtype=np.uint8)
    y_pred = np.zeros(y_true.shape, dtype=np.double)
    y_acc = np.zeros(y_true.shape, dtype=np.int)

    # topology
    tile_topology = roi.topology(width=tile_size, height=tile_size, overlap=overlap)

    # dataset and loader
    dataset = TileTopologyDataset(tile_topology, trans=in_trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_jobs)

    for ids, x in dataloader:
        x = x.to(device)
        y = model.forward(x, sigmoid=True)

        # accumulate predictions
        for i, identifier in enumerate(ids):
            x_off, y_off = tile_topology.tile_offset(identifier)
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
