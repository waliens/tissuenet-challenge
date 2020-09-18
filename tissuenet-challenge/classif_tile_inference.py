import numpy as np
import pyvips
from PIL import Image
import cv2
from shapely.affinity import affine_transform
from shapely.geometry import box
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from assets.sldc.locator import mask_to_objects_2d
from assets.sldc_pyvips.adapter import PyVipsTileBuilder, PyVipsSlide


class ExcludingEmptySlideDataset(Dataset):
    """A slide + a polygon : only provide images of tiles that intersects with the polygon."""
    def __init__(self, topology, tissue_poly, trans=None):
        self._topology = topology
        self._tissue_poly = tissue_poly
        self._filtered_identifiers = self._prepare(topology, tissue_poly)
        self._trans = trans

    @staticmethod
    def _prepare(topology, polygon):
        filtered2full_ids = list()
        for tile in topology:
            x, y = tile.abs_offset
            b = box(x, y, x + tile.width, y + tile.height)
            if b.intersects(polygon):
                filtered2full_ids.append(tile.identifier)
        return filtered2full_ids

    def __getitem__(self, item):
        image = Image.fromarray(self._topology.tile(item + 1).np_image)
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
        self._datasets = [ExcludingEmptySlideDataset(topology, tissue, trans=trans) for topology, tissue in zip(self._topologies, tissues)]
        self._sizes, self._cumsum_sizes = datasets_size_cumsum(self._datasets)

    def __getitem__(self, index):
        dataset_index, relative_index = get_sample_indexes(index, self._cumsum_sizes)
        return self._datasets[dataset_index][relative_index]

    def __len__(self):
        return self._cumsum_sizes[-1] + len(self._datasets[-1])


def foreground_detect(slide_path, zoom_level=0, tresh_c=5, morph_iter=3, area_ratio=2.5):
    vips_image = pyvips.Image.new_from_file(slide_path, page=zoom_level)
    height, width, bands = vips_image.height, vips_image.width, vips_image.bands
    image = np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, bands)
    )

    max_dim = max(height, width)
    block_size = int(0.45 * max_dim)
    block_size -= block_size % 2  # to make it odd

    thresh = cv2.adaptiveThreshold(image, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, tresh_c)
    kernel_dim = max(int(0.03 * max_dim), 5)
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=morph_iter)
    dilated = cv2.dilate(eroded, kernel, iterations=morph_iter)
    extended = cv2.copyMakeBorder(dilated, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=1)

    objects = mask_to_objects_2d(extended)

    # Only keep components greater than 2.5% of whole image
    min_area = int(area_ratio * width * height / 100)
    return [
      affine_transform(p, [2 ** zoom_level, 0, 0, 2 ** zoom_level, 0, 0])
      for p, _ in objects
      if p.area > min_area
    ]


def classify(slide_path, model, tile_size, num_workers=0, device="cpu", zoom_for_classif=2, zoom_for_filter=2, n_classes=4):
    # preprocessing
    tissues = foreground_detect(slide_path, zoom_level=zoom_for_filter)
    zoom_ratio = 1 / 2 ** zoom_for_classif
    tissues = [affine_transform(p, [zoom_ratio, 0, 0, zoom_ratio]) for p in tissues]

    # inference
    slide = PyVipsSlide(slide_path, zoom_level=zoom_for_classif)
    tile_builder = PyVipsTileBuilder(slide)
    dataset = MultiPolygonFilteredTopologyDataset(slide, tile_builder, tissues, trans=transform, max_width=tile_size, max_height=tile_size, overlap=0)
    loader = DataLoader(dataset, num_workers=num_workers)

    probas = np.zeros([len(dataset), n_classes])
    index = 0
    for _, tiles in loader:
        n_samples = int(tiles.size(0))
        probas[index:(index+n_samples)] = model.forward(tiles.to(device))
        index += n_samples

    classes = np.argmax(probas, axis=1)
    return np.max(classes)
