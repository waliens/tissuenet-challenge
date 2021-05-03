import os

from assets.inference import foreground_detect, determine_tissue_extract_level
import pyvips
import numpy as np
import cv2
from matplotlib import pyplot as plt


def plot(hist, prefix):
    plt.figure()
    plt.plot(np.arange(hist.shape[0] - 1), hist[:-1])
    plt.savefig(prefix + "-hist-no255.png")
    plt.close()
    plt.figure()
    plt.plot(np.arange(hist.shape[0]), hist)
    plt.savefig(prefix + "-hist-255.png")
    plt.close()
    plt.figure()
    plt.plot(np.arange(250), hist[:250])
    plt.savefig(prefix + "-hist-upto250.png")
    plt.close()


def segment(image, v_min=5, v_max=250):
    # get rid of white and black pixels in histogram
    bins, edges = np.hist(image.flatten(), bins=v_max - v_min, range=(v_min, v_max))
    edges = edges.astype(np.int)  # should be int
    # identify grey peak
    largest_bin = np.argmax(bins)
    largest_edge = edges[largest_bin]





def main(argv):
    images = list()
    hist_path = "/scratch/users/rmormont/tissuenet/hist"
    os.makedirs(hist_path, exist_ok=True)
    wsi_path = "/scratch/users/rmormont/tissuenet/wsis"
    for slide_path in os.listdir(wsi_path):
        wsi_full_path = os.path.join(wsi_path, slide_path)
        zoom_level = determine_tissue_extract_level(wsi_full_path, desired_processing_size=2048)
        vips_image = pyvips.Image.new_from_file(wsi_full_path, page=zoom_level)
        height, width, bands = vips_image.height, vips_image.width, vips_image.bands
        image = np.ndarray(
            buffer=vips_image.write_to_memory(),
            dtype=np.uint8,
            shape=(height, width, bands)
        )
        image = np.mean(image, axis=2).astype(np.uint8)  # grayscale
        images.append(image)

        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        filename = os.path.basename(slide_path).rsplit(".", 1)[0]
        cv2.imwrite(os.path.join(hist_path, filename + "-wsi.png"), image)
        plot(hist, os.path.join(hist_path, filename))

    hist = cv2.calcHist(images, [0], None, [256], [0, 256])
    plot(hist, os.path.join(hist_path, "00-all"))


    # tissues = foreground_detect(slide_path, fg_detect_rescale_to=2048, threshold=230)



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
