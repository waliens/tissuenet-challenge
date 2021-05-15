import os
from abc import abstractmethod

import PIL
import cv2
import math
import numpy as np
import sldc
from PIL import Image
from rasterio.features import rasterize
from shapely import wkt
from shapely.affinity import translate, affine_transform
from shapely.geometry import box
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
    def __init__(self, wsi, annotation, working_path, tile_size=512, zoom_level=0, n_jobs=0):
        self._annotation = annotation
        self._tile_size = tile_size
        self._wsi = CytomineSlide(wsi, zoom_level=zoom_level)
        self._builder = CytomineTileBuilder(working_path, n_jobs=n_jobs)
        self._working_path = working_path
        self._zoom_level = zoom_level

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

    def download(self):
        print("download '{}'".format(self._get_image_filepath()))
        return self._download_image()

    def _polygon(self):
        polygon = wkt.loads(self._annotation.location)
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

        translated = translate(self._polygon(), xoff=-(x_min + x), yoff=-(y_min + y))
        in_window = box(0, 0, self._tile_size, self._tile_size).intersection(translated)
        if in_window.is_empty:
            mask = np.zeros((self._tile_size, self._tile_size), dtype=np.uint8)
        else:
            mask = rasterize([in_window], out_shape=(self._tile_size, self._tile_size), fill=0, dtype=np.uint8) * 255

        return (x, y, self._tile_size, self._tile_size), crop, Image.fromarray(mask.astype(np.uint8))

    def crop_and_mask(self):
        """in image coordinates system, get full crop and mask"""
        (x_min, y_min), width, height = self._extract_image_box()
        image = self._robust_load_image()
        in_window = translate(self._polygon(), xoff=-x_min, yoff=-y_min)
        mask = rasterize([in_window], out_shape=(height, width), fill=0, dtype=np.uint8) * 255
        return image, mask

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
    def __init__(self, crop: BaseAnnotationCrop, cue):
        """
        Parameters
        ----------
        crop: BaseAnnotationCrop
        cue: ndarray
            Probability map for the cue np.array of float in [0, 1]
        """
        self._crop = crop
        self._cue = (cue * 255)

    def random_crop_and_mask(self):
        crop_location, crop, mask = self._crop.random_crop_and_mask()
        (x, y), w, h = crop_location
        final_mask = self._cue[y:(y+h), x:(x+w)]
        final_mask[mask > 0] = 255
        return crop_location, crop, final_mask

    def crop_and_mask(self):
        crop, mask = self.crop_and_mask()
        final_mask = self._cue
        final_mask[mask > 0] = 255
        return crop, final_mask


class RemoteAnnotationCropTrainDataset(Dataset):
    def __init__(self, crops, visual_trans=None, struct_trans=None):
        self._crops = crops
        self._stuct_trans = struct_trans
        self._visual_trans = visual_trans

    def __getitem__(self, item):
        annotation_crop = self._crops[item]
        _, image, mask = annotation_crop.random_crop_and_mask()

        if self._stuct_trans is not None:
            image, mask = self._stuct_trans([image, mask])
            mask = transforms.ToTensor()(mask)
        if self._visual_trans is not None:
            image = self._visual_trans(image)

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


def predict_roi(roi, ground_truth, model, device, in_trans=None, batch_size=1, tile_size=256, overlap=0, n_jobs=1, zoom_level=0):
    """
    Parameters
    ----------
    roi: AnnotationCrop
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
    ground_truth = [convert_poly(wkt.loads(g.location), zoom_level, roi.wsi.height) for g in ground_truth]
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
