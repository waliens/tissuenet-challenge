
import os
from collections import defaultdict

import numpy as np
from shapely.affinity import affine_transform
from skimage.io import imread
from sldc.locator import mask_to_objects_2d

from dataset import DatasetsGenerator, MemoryCrop, CropTrainDataset


def change_referential(polygon, height):
    return affine_transform(polygon, [1, 0, 0, -1, 0, height])


class SegpcDatasetGenerator(DatasetsGenerator):
    def __init__(self, data_path, tile_size, mask_folder="masks", image_folder="images", incomplete_folder="incomplete",
                 complete_folder="complete", missing_seed=42, remove_ratio=0.0, n_complete=1, n_validation=0):
        self._missing_seed = missing_seed
        self._remove_ratio = remove_ratio
        self._n_complete = n_complete - n_validation
        self._data_path = os.path.join(data_path, "{}_{:0.4f}_{}".format(missing_seed, remove_ratio, n_complete))
        self._train_path = os.path.join(self._data_path, "train")
        self._test_path = os.path.join(self._data_path, "test")
        self._mask_folder = mask_folder
        self._image_folder = image_folder
        self._incomplete_folder = incomplete_folder
        self._complete_folder = complete_folder
        self._tile_size = tile_size
        self._n_validation = n_validation

        self._annots_per_image = defaultdict(list)
        test_mask_path = os.path.join(self._test_path, self._mask_folder)
        for filename in os.listdir(test_mask_path):
            mask = imread(os.path.join(test_mask_path, filename))
            objects = mask_to_objects_2d(mask)
            if len(objects) > 0:
                objects, _ = zip(*objects)
            self._annots_per_image[filename] = [change_referential(o, mask.shape[0]) for o in objects]

    def _crops(self, path):
        images, masks = list(), list()
        for image_filename in os.listdir(os.path.join(path, self._image_folder)):
            images.append(os.path.join(path, self._image_folder, image_filename))
            mask_filename = image_filename.rsplit(".", 1)[0] + ".bmp"
            masks.append(os.path.join(path, self._mask_folder, mask_filename))
        return [MemoryCrop(i, m, tile_size=self._tile_size) for i, m in zip(images, masks)]

    def sets(self):
        incomplete = self._crops(os.path.join(self._train_path, self._incomplete_folder))
        complete = self._crops(os.path.join(self._train_path, self._complete_folder))
        np.random.shuffle(complete)
        complete, validation = complete[self._n_validation:], complete[:self._n_validation]
        return incomplete, complete, self._crops(self._test_path), validation

    def iterable_to_dataset(self, iterable, **kwargs):
        return CropTrainDataset(iterable, **kwargs)

    def roi_foregrounds(self, val_roi):
        return self._annots_per_image[os.path.basename(val_roi.img_path)]

    def crop(self, identifier):
        # id is image original filename
        filepath = self._find_by_walk(identifier, os.path.join(self._train_path, self._complete_folder, self._image_folder))
        if filepath is None:
            filepath = self._find_by_walk(identifier, os.path.join(self._test_path, self._image_folder))
        if filepath is None:
            raise ValueError("cannot find file with name '{}'".format(identifier))
        mask_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), self._mask_folder,
                                 identifier.replace(".tif", ".png"))
        return MemoryCrop(filepath, mask_path, tile_size=self._tile_size)

    def _find_by_walk(self, query, dir):
        for dirpath, dirnames, filenames in os.walk(dir, topdown=False):
            for filename in filenames:
                if filename == query:
                    return os.path.join(dirpath, filename)
        return None

