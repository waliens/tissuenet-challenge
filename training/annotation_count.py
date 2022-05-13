import os
import re
import sys
from abc import abstractmethod
from collections import defaultdict
from math import ceil

from cytomine.models import ImageInstanceCollection, AnnotationCollection, PropertyCollection
from imageio import imread
from numpy import random
from sklearn.utils import check_random_state

from sldc.locator import mask_to_objects_2d

from monuseg import MONUSEG_PROJECT


class BaseDatasetCounter(object):
    @abstractmethod
    def count(self, ms, nc, rr, **kwargs):
        pass


class MonusegDatasetCounter(BaseDatasetCounter):
    def __init__(self):
        self._images = ImageInstanceCollection().fetch_with_filter("project", MONUSEG_PROJECT)
        self._annotations = AnnotationCollection(project=MONUSEG_PROJECT, showWKT=True, showMeta=True).fetch()

        self._simages = sorted(self._images, key=lambda i: i.id)
        train, test = dict(), dict()

        for image in self._simages:
            properties = PropertyCollection(image).fetch().as_dict()
            if properties["set"].value == "train":
                train[image.id] = image
            else:
                test[image.id] = image

        self._train = train
        self._test = test

    def count(self, ms, nc, rr, **kwargs):
        random_state = check_random_state(ms)

        annot_per_image = defaultdict(list)
        # sorting for reproducibility
        for annot in sorted(self._annotations, key=lambda a: a.id):
            annot_per_image[annot.image].append(annot)

        train_ids = list(self._train.keys())
        random_state.shuffle(train_ids)
        incomplete = set(train_ids[nc:])

        kept_per_image, removed_per_image = defaultdict(lambda: 0), defaultdict(lambda: 0)

        for image in self._simages:
            if image.id not in train_ids:
                continue
            if image.id in incomplete:
                image_annots = annot_per_image[image.id]
                random_state.shuffle(image_annots)
                n_annots = len(image_annots)
                to_keep = ceil(n_annots * (1 - rr))
                kept_per_image[image.originalFilename] += to_keep
                removed_per_image[image.originalFilename] += n_annots - to_keep
            else:
                kept_per_image[image.originalFilename] = len(annot_per_image[image.id])
                removed_per_image[image.originalFilename] = 0

        kept = sum(kept_per_image.values())
        removed = sum(removed_per_image.values())

        return {
            "kept": kept,
            "removed": removed,
            "total": kept + removed,
            "per_image": {k: (v, removed_per_image[k]) for k, v in kept_per_image.items()}
        }


class GlasDatasetCounter(BaseDatasetCounter):
    def __init__(self, raw_path, min_area=350):
        test, train = set(), set()
        for filename in os.listdir(raw_path):
            if not filename.endswith(".bmp") or filename.endswith("anno.bmp"):
                continue
            filepath = os.path.join(raw_path, filename)
            if filename.startswith("train"):
                train.add(filepath)
            else:
                test.add(filepath)

        annotations = {}
        print("extract objects:")
        for train_image_path in train:
            print(">", train_image_path)
            mask = imread(self._get_mask_path(train_image_path))
            annotations[train_image_path] = [polygon for polygon, _ in mask_to_objects_2d(mask)]

        self._raw_path = raw_path
        self._annotations = annotations
        self._train = train
        self._test = test
        self._min_area = min_area

    @staticmethod
    def _get_mask_path(image_path):
        splitted = image_path.rsplit(".", 1)
        splitted.insert(1, "_anno.")
        return "".join(splitted)

    def count(self, ms, nc, rr, **kwargs):
        rng_generator = kwargs.get("rng_generator")

        if rng_generator is None:
            seed = 42
            rng = random.default_rng(seed)
            gen_ms = seed
        else:
            rng = random.default_rng(rng_generator)
            gen_ms = rng.integers(999999999, size=1)[0]

        if gen_ms != ms:
            raise ValueError("invalid randomization")

        # complete set
        complete_set = set(rng.choice(sorted(self._train), nc, replace=False))

        # incomplete set
        incomplete_set = self._train.difference(complete_set)
        large, small = [], []
        # !! NOT REPRODUCIBLE
        for filepath in incomplete_set:
            for annotation in self._annotations[filepath]:
                (large, small)[annotation.area <= self._min_area].append((filepath, annotation))

        filtered_large = rng.choice(large, int((1 - rr) * len(large)), replace=False)
        filtered_annotations = defaultdict(list)
        for filepath, polygon in [*filtered_large, *small]:
            filtered_annotations[filepath].append(polygon)

        kept = len(filtered_large)
        total = len(large)

        return {
            "kept": kept,
            "removed": total - kept,
            "total": total,
            # "per_image": {
            #     os.path.basename(fpath): (
            #         len(filtered_annotations[fpath]) if fpath in incomplete_set else len([p for p in polygons if p.area > self._min_area]),
            #         len([p for p in polygons if p.area > self._min_area]) - len(filtered_annotations[fpath]) if fpath in incomplete_set else 0
            #     )
            #     for fpath, polygons in self._annotations.items()
            # }
        }


class SegPcDatasetCounter(BaseDatasetCounter):
    def __init__(self, raw_dir):
        self._raw_dir = raw_dir
        self._files = self._extract_file_by_index(os.path.join(self._raw_dir, "train"))
        self._indexes, self._x_files, self._y_files_for_x = zip(*self._files)

    @staticmethod
    def _extract_file_by_index(folder):
        x_folder = os.path.join(folder, "x")
        y_folder = os.path.join(folder, "y")
        x_files = {
            int(x.rsplit(".", 1)[0][-4:]): os.path.join(x_folder, x)
            for x in os.listdir(x_folder)
            if x.endswith(".bmp")
        }

        y_files = os.listdir(y_folder)
        y_by_file = defaultdict(list)
        for index, x_filepath in x_files.items():
            pattern = re.compile(r"^" + str(index) + "_[0-9]+.bmp$")
            y_by_file[index] = [os.path.join(y_folder, filename)
                                for filename in y_files if pattern.match(filename) is not None]
            if len(y_by_file[index]) == 0:
                print("no match for x='{}'".format(x_filepath))

        return [(index, x_files[index], y_by_file[index]) for index in x_files.keys()]

    def count(self, ms, nc, rr, **kwargs):
        rng_generator = kwargs.get("rng_generator")

        if rng_generator is None:
            seed = 42
            rng = random.default_rng(seed)
            gen_ms = seed
        else:
            rng = random.default_rng(rng_generator)
            gen_ms = rng.integers(999999999, size=1)[0]

        if gen_ms != ms:
            print("missing seed mismatch (exp: {}, act: {}) so use old random state".format(ms, gen_ms), sys.stderr)
            rng = check_random_state(ms)

        complete_set = set(rng.choice(self._indexes, nc, replace=False))
        annotations_incomplete = [
            file
            for index, y_files in zip(self._indexes, self._y_files_for_x) if index not in complete_set
            for file in y_files]
        missing_set = set(rng.choice(annotations_incomplete, int(len(annotations_incomplete) * rr), replace=False))

        kept_per_image = defaultdict(lambda: 0)
        removed_per_image = defaultdict(lambda: 0)

        for i, (index, x_filepath, y_files) in enumerate(self._files):
            total = len(y_files)
            x_key = os.path.basename(x_filepath)
            if index in complete_set:
                kept_per_image[x_key] = total
            else:
                kept_per_image[x_key] = len([file for file in y_files if file not in missing_set])

            removed_per_image[x_key] = total - kept_per_image[x_key]

        kept = sum(kept_per_image.values())
        removed = sum(removed_per_image.values())

        return {
            "kept": kept,
            "removed": removed,
            "total": kept + removed,
            "per_image": {k: (v, removed_per_image[k]) for k, v in kept_per_image.items()}
        }



