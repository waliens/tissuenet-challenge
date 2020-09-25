import numpy as np
import pyvips
import torch
from PIL import Image
import cv2
from skimage.filters import threshold_otsu
from shapely.affinity import affine_transform
from shapely.geometry import box
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from assets.sldc.image import FixedSizeTileTopology
from assets.sldc.locator import mask_to_objects_2d
from assets.sldc_pyvips.adapter import PyVipsTileBuilder, PyVipsSlide


def check_tile_poly_intersection(tile, polygon):
    """A slide + a polygon : only provide images of tiles that intersects with the polygon."""
    x, y = tile.abs_offset
    b = box(x, y, x + tile.width, y + tile.height)
    return b.intersects(polygon)


def check_tile_variation(tile, max_mean, min_std):
    array = np.mean(tile.np_image, axis=2)
    return np.mean(array) < max_mean and np.std(array) > min_std


class TileExclusionDataset(Dataset):
    def __init__(self, topology, check_fn, *fn_args, trans=None, **fn_kwargs):
        self._topology = topology
        self._fn_args = fn_args
        self._fn_kwargs = fn_kwargs
        self._check_fn = check_fn
        self._trans = trans
        self._filtered_identifiers = self._prepare()

    def _prepare(self):
        filtered2full_ids = list()
        for tile in self._topology:
            if self._check_fn(tile, *self._fn_args, **self._fn_kwargs):
                filtered2full_ids.append(tile.identifier)
        return filtered2full_ids

    def __getitem__(self, item):
        image = Image.fromarray(self._topology.tile(self._filtered_identifiers[item]).np_image)
        if self._trans is not None:
            image = self._trans(image)
        return item + 1, image

    def __len__(self):
        return len(self._filtered_identifiers)


def datasets_size_cumsum(datasets):
    sizes = np.array([len(d) for d in datasets])
    cumsum = np.concatenate([np.array([0]), np.cumsum(sizes[:-1], dtype=np.int)])
    return sizes, cumsum


def get_sample_indexes(index, cumsum):
    dataset_index = np.searchsorted(cumsum, index, side="right") - 1
    relative_index = index - cumsum[dataset_index]
    return dataset_index, relative_index


class MultiPolygonFilteredTopologyDataset(Dataset):
    """A slide + several polygons: each polygon gets its own dataset (see ExcludingEmptyTilesDataset abobe). All are
    merged in this MultiPolygonFilteredTopologyDataset"""
    def __init__(self, slide, builder, tissues, trans=None, max_width=512, max_height=512, overlap=0):
        self._tissues = tissues
        self._topologies = [slide.window_from_polygon(tissue).tile_topology(builder, max_width=max_width, max_height=max_height, overlap=overlap) for tissue in tissues]
        # filter topology not larger/wider than the tile size
        self._topologies, self._tissues = zip(*[
            (topology, tissue)
            for topology, tissue in zip(self._topologies, self._tissues)
            if topology._image.width >= max_width and topology._image.height >= max_height
        ])
        self._topologies = [FixedSizeTileTopology(t) for t in self._topologies]
        self._datasets = [
            TileExclusionDataset(topology, check_tile_poly_intersection, tissue, trans=trans)
            for topology, tissue in zip(self._topologies, self._tissues)
        ]
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        return self._datasets[dataset_index][relative_index]

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])


class VariationFilteredTopologyDataset(Dataset):
    def __init__(self, slide, builder, mean=210, std=5, max_width=512, max_height=512, overlap=0):
        self._mean = mean
        self._std = std




def get_image_meta(path):
    """
    n-pages, (height, width)
    """
    image = pyvips.Image.new_from_file(path)
    return image.get("n-pages"), (image.height, image.width)


def determine_tissue_extract_level(slide_path, desired_processing_size=2048):
    levels, (max_height, max_width) = get_image_meta(slide_path)
    ref_size = max(max_width, max_height)
    best_size = ref_size
    best_level = 0
    while best_size > desired_processing_size and best_level < levels - 1:
        best_size //= 2
        best_level += 1
    return best_level


def foreground_detect(slide_path, fg_detect_rescale_to=2048, morph_iter=3, area_ratio=0.005):
    zoom_level = determine_tissue_extract_level(slide_path, desired_processing_size=fg_detect_rescale_to)
    vips_image = pyvips.Image.new_from_file(slide_path, page=zoom_level)
    height, width, bands = vips_image.height, vips_image.width, vips_image.bands
    image = np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, bands)
    )
    image = np.mean(image, axis=2).astype(np.uint8)  # grayscale

    max_dim = max(height, width)
    extr_mask = np.logical_and(image > 75, image < 250)
    threshold = threshold_otsu(image[extr_mask])
    # remove extremum
    # also remove black pixels
    thresh = (image <= threshold).astype(np.uint8)
    kernel_dim = max(int(0.005 * max_dim), 3)
    kernel_dim -= 1 - kernel_dim % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(kernel_dim, kernel_dim))
    dilated = cv2.dilate(thresh, kernel, iterations=morph_iter)
    eroded = cv2.erode(dilated, kernel, iterations=morph_iter)

    objects = mask_to_objects_2d(eroded)

    # Only keep components greater than 2.5% of whole image
    min_area = int(area_ratio * width * height / 100)
    filtered = [p for p, _ in objects if p.area > min_area]

    if len(filtered) == 0:
        return []

    # merge intersecting polygons
    to_check_for_merge = list(filtered)
    checked_for_merge = list()
    while len(to_check_for_merge) > 0:
        poly = to_check_for_merge.pop(0)
        merged = False
        for i in range(len(to_check_for_merge)):
            if poly.intersects(to_check_for_merge[i]):
                to_check_for_merge[i] = to_check_for_merge[i].union(poly)
                merged = True
                break
        if not merged:
            checked_for_merge.append(poly)

    def filter_by_shape(p):
        """Filter polygons that are at least 20 higher (resp wider) than wide (resp. high)
        Ok
        """
        xmin, ymin, xmax, ymax = p.bounds
        width = xmax - xmin
        height = ymax - ymin
        if width > height:
            return (width / height) < 20
        else:
            return (height / width) < 20

    return [
       affine_transform(p, [2 ** zoom_level, 0, 0, 2 ** zoom_level, 0, 0])
       for p in checked_for_merge if filter_by_shape(p)
    ]


def classify(slide_path, model, device, transform, batch_size=16, tile_size=512, tile_overlap=0, num_workers=0, zoom_level=2, n_classes=4, fg_detect_rescale_to=2048):
    # preprocessing
    tissues = foreground_detect(slide_path, fg_detect_rescale_to=fg_detect_rescale_to)
    zoom_ratio = 2 ** zoom_level
    tissues = [affine_transform(p, [1 / zoom_ratio, 0, 0, 1 / zoom_ratio, 0, 0]) for p in tissues]

    if len(tissues) == 0:
        raise ValueError("no poly for slide '{}'".format(slide_path))

    print("--- {} ---".format(slide_path))
    slide = PyVipsSlide(slide_path, zoom_level=zoom_level)
    tile_builder = PyVipsTileBuilder(slide)
    dataset = MultiPolygonFilteredTopologyDataset(
        slide, tile_builder, tissues, trans=transform, max_width=tile_size, max_height=tile_size,
        overlap=tile_overlap)

    # inference
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    print("Number of areas to process: {}".format(len(tissues)))
    print("Number of tiles to process: {}".format(len(dataset)))

    probas = np.zeros([len(dataset), n_classes])
    index = 0
    for _, tiles in loader:
        tiles = tiles.to(device)
        n_samples = int(tiles.size(0))
        probas[index:(index+n_samples)] = torch.nn.functional.softmax(model.forward(tiles), dim=1).detach().cpu().numpy()
        index += n_samples

    classes = np.argmax(probas, axis=1)
    print("class_dict: {}".format({v: c for v, c in zip(*np.unique(classes, return_counts=True))}))
    print(">> predicting: {}".format(np.max(classes)))
    return np.max(classes)
