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
            filenames.append(row[0])
        return filenames


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device", default="cuda:0")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=5, type=int)
    args, _ = parser.parse_known_args(argv)

    ZOOM_LEVEL = 2
    N_CLASSES = 4
    TILE_SIZE = 320
    TILE_OVERLAP = 0
    BATCH_SIZE = 16
    ARCH = "densenet121"
    MODEL_PATH = os.path.join("assets", "densenet121_mtdp_e_10_val_0.8000_sco_0.9464_z2_1600972592.605832.pth")

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    device = torch.device(args.device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    features = build_model(arch=ARCH, pretrained=False, pool=True)
    model = SingleHead(features, Head(features.n_features(), n_classes=4))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    results = {}
    for filename in read_test_files():
        with torch.no_grad():
            results[filename] = classify(
                slide_path=os.path.join("data", filename),
                model=model,
                device=device,
                transform=trans,
                batch_size=BATCH_SIZE,
                tile_size=TILE_SIZE,
                tile_overlap=TILE_OVERLAP,
                num_workers=args.n_jobs - 1,
                zoom_level=ZOOM_LEVEL,
                n_classes=N_CLASSES
            )

    write_submission(results)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
