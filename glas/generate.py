import itertools
import os
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

from imageio import imsave
from joblib import Parallel, delayed
from numpy import random
from numpy.random import RandomState, SeedSequence
from rasterio.features import rasterize
from skimage.io import imread
from skimage.transform import resize, rescale
from sldc.locator import mask_to_objects_2d


def copy_images_and_masks(dest_dir, images):
    image_dir = os.path.join(dest_dir, "images")
    mask_dir = os.path.join(dest_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for filepath in images:
        filename = os.path.basename(filepath)
        shutil.copy(filepath, image_dir)
        shutil.copy(get_mask_path(filepath), os.path.join(mask_dir, filename))


def generate_one(train, tmp_test_folder, annotations, rr, nc, outdir, rng_generator=None, min_area=350):
    if rng_generator is None:
        seed = 42
        rng = random.default_rng(seed)
        ms = seed
    else:
        rng = random.default_rng(rng_generator)
        ms = rng.integers(999999999, size=1)[0]
    folder = os.path.join(outdir, "{}_{:0.4f}_{}".format(ms, rr, nc))
    if os.path.exists(folder):
        return
    train_folder = os.path.join(folder, "train")

    # complete set
    complete_folder = os.path.join(train_folder, "complete")
    complete_set = set(rng.choice(sorted(train), nc, replace=False))
    copy_images_and_masks(complete_folder, complete_set)

    # test set
    test_folder = os.path.join(folder, "test")
    shutil.copytree(tmp_test_folder, test_folder, dirs_exist_ok=True)

    # incomplete set
    incomplete_set = train.difference(complete_set)
    large, small = [], []
    for filepath in incomplete_set:
        for annotation in annotations[filepath]:
            (large, small)[annotation.area <= min_area].append((filepath, annotation))

    filtered_large = rng.choice(large, int((1 - rr) * len(large)), replace=False)
    filtered_annotations = defaultdict(list)
    for filepath, polygon in [*filtered_large, *small]:
        filtered_annotations[filepath].append(polygon)

    incomplete_folder = os.path.join(train_folder, "incomplete")
    incomplete_image_folder = os.path.join(incomplete_folder, "images")
    incomplete_mask_folder = os.path.join(incomplete_folder, "masks")
    os.makedirs(incomplete_image_folder, exist_ok=True)
    os.makedirs(incomplete_mask_folder, exist_ok=True)
    for filepath in incomplete_set:
        filename = os.path.basename(filepath)
        shutil.copy(filepath, incomplete_image_folder)
        fg = filtered_annotations[filepath]
        image = imread(filepath)
        h, w = image.shape[:2]
        if len(fg) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = rasterize(fg, out_shape=(h, w), fill=0, dtype=np.uint8) * 255
        imsave(os.path.join(incomplete_mask_folder, filename), mask)


def get_mask_path(image_path):
    splitted = image_path.rsplit(".", 1)
    splitted.insert(1, "_anno.")
    return "".join(splitted)


def main(argv):
    argparse = ArgumentParser()
    argparse.add_argument("--dir", "-d", dest="dir")
    argparse.add_argument("--outdir", "-o", dest="outdir")
    params, _ = argparse.parse_known_args(args=argv)

    test_set, train_set = set(), set()
    for filename in os.listdir(params.dir):
        if not filename.endswith(".bmp") or filename.endswith("anno.bmp"):
            continue
        filepath = os.path.join(params.dir, filename)
        if filename.startswith("train"):
            train_set.add(filepath)
        else:
            test_set.add(filepath)

    train_annotations = {}
    print("extract objects:")
    for train_image_path in train_set:
        print(">", train_image_path)
        mask = imread(get_mask_path(train_image_path))
        train_annotations[train_image_path] = [polygon for polygon, _ in mask_to_objects_2d(mask)]

    glas_rr = [1.0, 0.99, 0.975, 0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.5, 0.25]
    glas_nc = [2, 4, 8, 16, 24, 32, 40, 60]
    rngs = random.SeedSequence(42).spawn(10)

    with TemporaryDirectory() as tempdir:
        tmp_test_dir = os.path.join(tempdir, "test")
        copy_images_and_masks(tmp_test_dir, test_set)

        generate_one(train_set, tmp_test_dir, train_annotations, 0.0, 85, params.outdir)

        for rr, nc, rng in itertools.product(glas_rr, glas_nc, rngs):
            if not ((0.89 < rr < 0.91) ^ (nc == 8)):
                continue
            print(rr, nc, rng)
            generate_one(train_set, tmp_test_dir, train_annotations, rr, nc, params.outdir, rng_generator=rng)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
