import itertools
import os
from numpy import random
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from collections import defaultdict
from argparse import ArgumentParser

from imageio import imsave
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.transform import resize, rescale

from stats import group_by

TERMS = {
    "cytoplasm": 543560059,  # 20
    "nucleus": 543560090  # 40
}


def readall(file):
    with open(file, "r") as f:
        return [a.strip() for a in f.readlines()]


def copy_update(filepath, dest, data_type="x", target_size=512, image=None):
    if image is None:
        image = imread(filepath).astype(np.uint8)

    if data_type == "y":
        image = (image > 0).astype(np.uint8) * 255

    h, w = image.shape[:2]

    out_shape = [target_size, int(w * target_size / h)]
    if h > w:
        out_shape = [int(h * target_size / w), target_size]

    out_image = resize(image, output_shape=out_shape, order=0 if data_type == "y" else 1, preserve_range=True).astype(np.uint8)

    if data_type == "y":
        out_image[out_image > 127] = 255
        out_image[out_image <= 127] = 0

    return imsave(os.path.join(dest, os.path.basename(filepath)), out_image)


def extract_file_by_index(dir):
    x_folder = os.path.join(dir, "x")
    y_folder = os.path.join(dir, "y")
    x_files = {
        int(x.rsplit(".", 1)[0][-4:]): os.path.join(x_folder, x)
        for x in os.listdir(x_folder)
        if x.endswith(".bmp")
    }

    y_files = os.listdir(y_folder)
    y_by_file = defaultdict(list)
    for index, x_filepath in x_files.items():
        pattern = re.compile(r"^" + str(index) + "_[0-9]+.bmp$")
        y_by_file[index] = [os.path.join(y_folder, filename) for filename in y_files if pattern.match(filename) is not None]
        if len(y_by_file[index]) == 0:
            print("no match for x='{}'".format(x_filepath))

    return [(index, x_files[index], y_by_file[index]) for index in x_files.keys()]


def load_mask(y_file):
    mask = imread(y_file)
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    return mask


def copy_val_set(dir, dest_dir, target_size=512, folder_name="validation"):
    files = extract_file_by_index(os.path.join(dir, folder_name))
    x_dest = os.path.join(dest_dir, "images")
    y_dest = os.path.join(dest_dir, "masks")

    os.makedirs(x_dest, exist_ok=True)
    os.makedirs(y_dest, exist_ok=True)

    for i, (index, x_filepath, y_files) in enumerate(files):
        #print("\r", 100 * (i+1) / len(files), "%", end="", flush=True)
        copy_update(x_filepath, x_dest, data_type="x", target_size=target_size)
        if len(y_files) > 0:
            mask = np.max(np.array([load_mask(y_file) for y_file in y_files]), axis=0)
        else:
            mask = np.zeros(imread(x_filepath).shape[:2], dtype=np.uint8)
        copy_update(x_filepath, y_dest, data_type="y", target_size=target_size, image=mask)
    #print()


def copy_train_set(dir, dest_dir, random_state, n_complete, missing_ratio, target_size=512, folder_name="train"):
    files = extract_file_by_index(os.path.join(dir, folder_name))

    indexes, x_files, y_files_for_x = zip(*files)

    dests = {
        ('x', 'complete'): os.path.join(dest_dir, 'complete', 'images'),
        ('x', 'incomplete'): os.path.join(dest_dir, 'incomplete', 'images'),
        ('y', 'complete'): os.path.join(dest_dir, 'complete', 'masks'),
        ('y', 'incomplete'): os.path.join(dest_dir, 'incomplete', 'masks')
    }

    for path in dests.values():
        os.makedirs(path, exist_ok=True)

    complete_set = set(random_state.choice(indexes, n_complete, replace=False))
    annotations_incomplete = [file for index, y_files in zip(indexes, y_files_for_x) if index not in complete_set for file in y_files]
    missing_set = set(random_state.choice(annotations_incomplete, int(len(annotations_incomplete) * missing_ratio), replace=False))

    for i, (index, x_filepath, y_files) in enumerate(files):
        #print("\r", 100 * (i+1) / len(files), "%", end="", flush=True)
        if index in complete_set:
            _type = 'complete'
            kept_annotations = y_files
        else:
            _type = 'incomplete'
            kept_annotations = [file for file in y_files if file not in missing_set]

        copy_update(x_filepath, dests[('x', _type)], data_type='x', target_size=target_size)
        if len(kept_annotations) > 0:
            mask = np.max(np.array([load_mask(y_file) for y_file in kept_annotations]), axis=0)
        else:
            mask = np.zeros(imread(x_filepath).shape[:2], dtype=np.uint8)
        copy_update(x_filepath, dests[('y', _type)], data_type="y", target_size=target_size, image=mask)
    #print()


def main(argv):
    argparse = ArgumentParser()
    argparse.add_argument("--dir", "-d", dest="dir")
    argparse.add_argument("--outdir", "-o", dest="outdir")
    params, _ = argparse.parse_known_args(args=argv)

    n_complete = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    missing_ratio = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.60, 0.5, 0.25]

    valid_combs = [(nc, rr) for nc, rr in itertools.product(n_complete, missing_ratio) if (0.89 < rr < 0.91) and nc != 30]
    print(valid_combs)
    print("number of comb == {}".format(len(valid_combs)))
    target_dim = 512
    rngs = random.SeedSequence(42).spawn(10)

    with TemporaryDirectory() as dirname:
        val_to_copy = os.path.join(dirname, "validation")
        copy_val_set(params.dir, val_to_copy, target_size=target_dim)

        # full_folder = os.path.join(params.outdir, "42_0.0000_298")
        # copy_val_set(params.dir, os.path.join(full_folder, "validation"), target_size=target_dim)
        # copy_val_set(params.dir, dest_dir=os.path.join(full_folder, "train", "complete"), target_size=target_dim,
        #              folder_name="train")

        def process(nc, mr, rng_generator=None):
            if rng_generator is None:
                seed = 42
                rng = random.default_rng(seed)
                ms = seed
            else:
                rng = random.default_rng(rng_generator)
                ms = rng.integers(999999999, size=1)[0]
            print(nc, mr, ms)
            gen_dir = os.path.join(params.outdir, "{}_{:1.4f}_{}".format(ms, mr, nc))
            if os.path.exists(gen_dir):
                return
            shutil.copytree(val_to_copy, os.path.join(gen_dir, "validation"))
            copy_train_set(params.dir, os.path.join(gen_dir, "train"),
                           random_state=rng, n_complete=nc, missing_ratio=mr, target_size=target_dim)

        Parallel(n_jobs=10)(delayed(process)(nc, rr, rngs[j]) for nc, rr in valid_combs for j in range(10))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
