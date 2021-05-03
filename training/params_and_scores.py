import numpy as np
from clustertools import build_datacube


def main(argv):
    cube = build_datacube("thyroid-unet-training")

    for (z, s), cube in cube.iter_dimensions("zoom_level", "tile_size"):
        print(z, s, np.mean(cube("val_losses")))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
