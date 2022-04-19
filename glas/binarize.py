import os
import shutil
from argparse import ArgumentParser

import numpy as np
from skimage.io import imread, imsave


def main(argv):
    argparse = ArgumentParser()
    argparse.add_argument("--dir", "-d", dest="dir")
    argparse.add_argument("--outdir", dest="outdir")
    params, _ = argparse.parse_known_args(args=argv)

    os.makedirs(params.outdir, exist_ok=True)

    for filename in os.listdir(params.dir):
        src = os.path.join(params.dir, filename)
        dst = os.path.join(params.outdir, filename)
        if "anno" not in filename:
            shutil.copyfile(src, dst)
        else:
            mask = imread(src)
            mask[mask > 0] = 1.0
            imsave(dst, mask.astype(np.uint8) * 255)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
