import logging
import os

import numpy as np
from PIL import Image
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from rasterio.features import rasterize
from shapely import wkt
from shapely.affinity import translate, affine_transform
from shapely.geometry import box
from sldc import TileTopology
from sldc.image import FixedSizeTileTopology
from sldc_openslide import OpenSlideTileBuilder, OpenSlideImage, OpenSlideTile
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


class RemoteAnnotationTrainDataset(Dataset):
    def __init__(self, collection, images, working_path, in_trans=None, seg_trans=None, width=256, height=256):
        self._collection = collection
        self._working_path = working_path
        self._width = width
        self._height = height
        self._images = images
        self._seg_trans = seg_trans
        self._in_trans = in_trans

    def __getitem__(self, item):
        annotation = self._collection[item]
        wsi = OpenSlideImage(self._images[annotation.image].download_path)
        width, height = self._width, self._height

        if wsi.height < height or wsi.width < width:
            raise ValueError("cannot extract patch, original wsi is too small")

        polygon = wkt.loads(annotation.location)
        polygon = affine_transform(polygon, [1, 0, 0, -1, 0, wsi.height])

        x_min, y_min, x_max, y_max = polygon.bounds
        crop_width, crop_height = x_max - x_min, y_max - y_min

        if crop_width <= width:
            x = x_min + (crop_width - width) / 2
        else:
            x = np.random.randint(x_min, x_max - width)

        if crop_height <= height:
            y = y_min + (crop_height - height) / 2
        else:
            y = np.random.randint(y_min, y_max - height)

        # annotation on image border
        x = int(min(x, wsi.width - width))
        y = int(min(y, wsi.height - height))

        # rasterize polygon
        translated = translate(polygon, xoff=-x, yoff=-y)
        in_window = box(0, 0, width, height).intersection(translated)
        if in_window.is_empty:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = rasterize([in_window], out_shape=(height, width), fill=0, dtype=np.uint8) * 255

        # transform
        mask = Image.fromarray(mask.astype(np.uint8))
        image = Image.fromarray(wsi.window((x, y), width, height).np_image)
        if self._seg_trans is not None:
            image, mask = self._seg_trans(image, mask)
        if self._in_trans is not None:
            image = self._in_trans(image)

        return image, mask

    def __len__(self):
        return len(self._collection)


class TileTopologyDataset(Dataset):
    def __init__(self, topology, trans=None, cyto_argv=None):
        self._topology = topology
        self._trans = trans
        self._cyto_argv = cyto_argv

    def __getitem__(self, item):
        self._check_cytomine_connected()
        image = Image.fromarray(self._topology.tile(item + 1).np_image)
        if self._trans is not None:
            image = self._trans(image)
        return item + 1, image

    def __len__(self):
        return len(self._topology)

    def _check_cytomine_connected(self):
        try:
            self._cytomine = Cytomine.get_instance()
        except ConnectionError:
            self._cytomine = Cytomine.connect_from_cli(self._cyto_argv)
        self._cytomine.logger.setLevel(logging.CRITICAL)


def predict_roi(image, roi, ground_truth, model, device, in_trans=None, batch_size=1, tile_size=256, overlap=0, n_jobs=1, working_path="/tmp", cyto_argv=None):
    """
    Parameters
    ----------
    image: sldc.Image
    roi: Annotation
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
    working_path: str
        Working path (temporary folder for storing downloaded _images)
    cyto_argv: list
        Argv for init cytomine client on worker thread

    Returns
    -------
    """
    # build ground truth
    affine_matrix = [1, 0, 0, -1, 0, image.height]
    roi_poly = affine_transform(wkt.loads(roi.location), affine_matrix)
    ground_truth = [affine_transform(wkt.loads(g.location), affine_matrix) for g in ground_truth]
    min_x, min_y, max_x, max_y = (int(v) for v in roi_poly.bounds)
    mask_dims = (int(max_x - min_x), int(max_y - min_y))
    translated_gt = [translate(g.intersection(roi_poly), xoff=-min_x, yoff=-min_y) for g in ground_truth]
    y_true = rasterize(translated_gt, out_shape=mask_dims, fill=0, dtype=np.uint8)
    y_pred = np.zeros(y_true.shape, dtype=np.double)
    y_acc = np.zeros(y_true.shape, dtype=np.int)

    # topology
    builder = OpenSlideTileBuilder()
    slide = OpenSlideImage(image.download_path)
    window = slide.window((min_x, min_y), mask_dims[0], mask_dims[1])
    base_topology = TileTopology(window, builder, max_width=tile_size, max_height=tile_size, overlap=overlap)
    tile_topology = FixedSizeTileTopology(base_topology)

    # dataset and loader
    dataset = TileTopologyDataset(tile_topology, trans=in_trans, cyto_argv=cyto_argv)
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
    # roi.dump("{}_image.png".format(roi.id), override=False)
    # cv2.imwrite("{}_true.png".format(roi.id), y_true * 255)
    # cv2.imwrite("{}_pred_{}.png".format(roi.id, datetime.now().timestamp()), (y_pred * 255).astype(np.uint8))
    return y_pred, y_true


def main(argv):
    with Cytomine.connect_from_cli(argv):
        collection = AnnotationCollection(
            project=77150529, showWKT=True, showMeta=True, showTerm=True).fetch()
        collection = collection.filter(lambda a: (a.user in {55502856} and len(a.term) > 0 and a.term[0] in {35777351, 35777321, 35777459}))
        dataset = RemoteAnnotationTrainDataset(
            collection,
            in_trans=transforms.Lambda(lambda t: t / 255.0),
            seg_trans=segmentation_transform,
            working_path="./tmp"
        )

        np.random.seed(42)

        if len(dataset) == 0:
            raise ValueError("no image in dataset")

        for i in range(len(dataset)):
            im, mask = dataset[i]
            im.save("im{}.png".format(i))
            mask.save("im{}_ma.png".format(i))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])