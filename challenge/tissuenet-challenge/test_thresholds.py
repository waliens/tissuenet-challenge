import os
import pyvips
from skimage.filters import try_all_threshold
import numpy as np
from assets.inference import determine_tissue_extract_level


def main(argv):
    wsi_path = "/scratch/users/rmormont/tissuenet/wsis"
    wsi_full_path = os.path.join(wsi_path, "C13_B324_S11.tif") #
    zoom_level = determine_tissue_extract_level(wsi_full_path, desired_processing_size=2048)
    vips_image = pyvips.Image.new_from_file(wsi_full_path, page=zoom_level)
    height, width, bands = vips_image.height, vips_image.width, vips_image.bands
    image = np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=(height, width, bands)
    )
    image = np.mean(image, axis=2).astype(np.uint8)  # grayscale

    figure = try_all_threshold(image, figsize=(10, 6), verbose=True)
    figure[0].savefig("all_thresholds.png", dpi=300)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
