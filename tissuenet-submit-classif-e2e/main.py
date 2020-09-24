import os
import csv
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import transforms

from assets.inference import classify
from assets.mtdp import build_model
from assets.mtdp.components import Head
from assets.mtdp.networks import SingleHead


def write_submission(preds):
    with open("submission.csv", "w+") as file:
        file.write("filename,0,1,2,3\n")
        for filename, pred_cls in preds.items():
            file.write(os.path.basename(filename) + "," + ",".join([str(int(pred_cls == cls)) for cls in range(4)]) + "\n")


def read_test_files():
    with open("data/test_metadata.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)
        filenames = list()
        for row in reader:
            filenames.append(os.path.join("assets", row[0]))
        return filenames


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device", default="cuda:0")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=5, type=int)
    args, _ = parser.parse_known_args(argv)

    ZOOM_LEVEL = 2
    N_CLASSES = 4
    TILE_SIZE = 320
    MODEL_PATH = os.path.join("assets", ".pth")

    trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ]

    device = torch.device(args.device)

    device = torch.device(args.device)
    model = torch.load(MODEL_PATH, map_location=device)

    for filename in read_test_files():
        cls = classify(
            slide_path=os.path.join("data", filename),
            model=model,
            device=device,
            transform=trans,
            tile_size=TILE_SIZE,
            num_workers=args.n_jobs - 1,
            zoom_level=ZOOM_LEVEL,
            n_classes=N_CLASSES
        )


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
