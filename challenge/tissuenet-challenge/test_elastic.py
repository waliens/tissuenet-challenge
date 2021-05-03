import os

import cv2
import numpy as np
from PIL import Image
from numpy.random.mtrand import RandomState

from e2e_classifier_train import elastic_transform


def main(argv):
    for filename in os.listdir("images"):
        if "alpha" in filename:
            continue
        path = os.path.join("images", filename)
        pil_img = Image.open(path).convert("RGB")
        img = np.asarray(pil_img.resize((pil_img.width // 4, pil_img.height // 4)))

        for alpha in [80, 100, 120, 140, 150]:
            for sigma in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
                result = elastic_transform(img, alpha=alpha, sigma=sigma, random_state=RandomState(42))
                save_path = os.path.join("images", filename.rsplit(".", 1)[0] + "_alpha={}_sigma={}_".format(alpha, sigma) + ".tif")
                print(save_path)
                Image.fromarray(result).resize((pil_img.width, pil_img.height)).save(save_path)
                # cv2.imwrite(, cv2.resize(result, (pil_img.height, pil_img.width)))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
