import itertools
import os
import random
from collections import defaultdict
from math import ceil

import cv2
import numpy as np
from PIL import Image
from cytomine import Cytomine
from cytomine.models import ImageInstanceCollection, AnnotationCollection, PropertyCollection
from rasterio.features import rasterize
from shapely import wkt
from sklearn.utils import check_random_state

from dataset import DatasetsGenerator, convert_poly, MemoryCrop, CropTrainDataset

MONUSEG_PROJECT = 532820586


def get_monuseg_data(data_path, mask_folder="masks", image_folder="images", incomplete_folder="incomplete",
                     complete_folder="complete", remove_ratio=0.0, seed=42, n_complete=2):
    random_state = check_random_state(seed)
    data_path = os.path.join(data_path, "{}_{:0.4f}_{}".format(seed, remove_ratio, n_complete))
    if os.path.exists(data_path):
        return
    print(data_path)
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    images = ImageInstanceCollection().fetch_with_filter("project", MONUSEG_PROJECT)
    annotations = AnnotationCollection(project=MONUSEG_PROJECT, showWKT=True, showMeta=True).fetch()

    annot_per_image = defaultdict(list)
    # sorting for reproducibility
    for annot in sorted(annotations, key=lambda a: a.id):
        annot_per_image[annot.image].append(annot)

    train, test = dict(), dict()
    # sorting for reproducibility
    simages = sorted(images, key=lambda i: i.id)
    for image in simages:
        properties = PropertyCollection(image).fetch().as_dict()
        if properties["set"].value == "train":
            train[image.id] = image
        else:
            test[image.id] = image

    train_ids = list(train.keys())
    random_state.shuffle(train_ids)
    incomplete = set(train_ids[n_complete:])

    for image in simages:
        if image.id in incomplete:
            write_path = os.path.join(train_path, incomplete_folder)
            image_annots = annot_per_image[image.id]
            random_state.shuffle(image_annots)
            n_annots = len(image_annots)
            to_keep = ceil(n_annots * (1 - remove_ratio))
            image_annots = image_annots[:to_keep]
        else:
            if image.id in train:
                write_path = os.path.join(train_path, complete_folder)
            else:
                write_path = test_path
            image_annots = annot_per_image[image.id]

        image_path = os.path.join(write_path, image_folder)
        os.makedirs(image_path, exist_ok=True)
        image.download(os.path.join(image_path, "{originalFilename}"), override=False)

        fg = [convert_poly(wkt.loads(a.location), 0, image.height) for a in image_annots]
        if len(fg) == 0:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            mask = rasterize(fg, out_shape=(image.height, image.width), fill=0, dtype=np.uint8) * 255
        mask_path = os.path.join(write_path, mask_folder)
        os.makedirs(mask_path, exist_ok=True)
        cv2.imwrite(os.path.join(mask_path, image.originalFilename.replace(".tif", ".png")), mask)


class MonusegDatasetGenerator(DatasetsGenerator):
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

        images = ImageInstanceCollection().fetch_with_filter("project", MONUSEG_PROJECT)
        annotations = AnnotationCollection(project=MONUSEG_PROJECT, showWKT=True, showMeta=True).fetch()
        a2i = defaultdict(list)
        for annot in annotations:
            a2i[annot.image].append(annot)
        self._annots_per_image = dict()
        for image in images:
            self._annots_per_image[image.originalFilename] = a2i[image.id]

    def _crops(self, path):
        images, masks = list(), list()
        for image_filename in os.listdir(os.path.join(path, self._image_folder)):
            images.append(os.path.join(path, self._image_folder, image_filename))
            mask_filename = image_filename.rsplit(".", 1)[0] + ".png"
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
        # if filepath is None:
        #     filepath = self._find_by_walk(identifier, os.path.join(self._train_path, self._incomplete_folder, self._image_folder))
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


def main(argv):
    with Cytomine.connect_from_cli(argv) as conn:
        np.random.seed(42)
        remove_ratios = [0.95, 0.85, 0.8, 0.75, 0.60, 0.5, 0.25, 0.99, 0.975]
        n_completes = [2] #, 10, 15]
        seeds = np.random.randint(0, 99999999, [10])
        #
        for remove_ratio, n_complete, seed in itertools.product(remove_ratios, n_completes, seeds):
            get_monuseg_data("/scratch/users/rmormont/monuseg",
                             remove_ratio=remove_ratio, n_complete=n_complete, seed=seed)

        # get_monuseg_data("/scratch/users/rmormont/monuseg",
        #                  remove_ratio=0, n_complete=30, seed=42)
        #
        # for seed in seeds:
        #     for n_complete in n_completes:
        #         get_monuseg_data("/scratch/users/rmormont/monuseg", remove_ratio=1.0, n_complete=n_complete, seed=seed)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
