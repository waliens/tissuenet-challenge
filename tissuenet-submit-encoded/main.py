import os
import csv
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import transforms

from assets.inference import classify, TimingContextManager
from assets.mtdp import build_model
from assets.mtdp.components import Head
from assets.mtdp.networks import SingleHead
from assets.slide_encoding import CustomSlideClassifier, ModelSlideEncoder


def write_submission(preds):
    with open("submission.csv", "w+") as file:
        file.write("filename,0,1,2,3\n")
        for filename, pred_cls in preds:
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
    TILE_SIZE = 320
    TILE_OVERLAP = 0
    BATCH_SIZE = 32
    ARCH = "densenet121"
    ENCODING_MODEL_PATH = os.path.join("assets", "densenet121-mh-best-191205-141200.pth")
    CLASSIFIER_MODEL_PATH = os.path.join("assets", "wsi_encoded_e_80_lr_0.001_val_0.9079_2020-10-28T00_10_17.607823.pth")

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    device = torch.device(args.device)

    tile_encoder = build_model(arch=ARCH, pretrained=False, pool=True)
    encoded_state_dict = torch.load(ENCODING_MODEL_PATH, map_location=device)
    tile_encoder.load_state_dict({k: v for k, v in encoded_state_dict.items() if "heads" not in k})
    tile_encoder.eval()
    tile_encoder.to(device)

    classifier_state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)
    slide_classifier = CustomSlideClassifier()
    slide_classifier.load_state_dict(classifier_state_dict)
    slide_classifier.eval()
    slide_classifier.to(device)

    encoder = ModelSlideEncoder(
        tile_encoder, trans=trans, tile_size=TILE_SIZE, tile_overlap=TILE_OVERLAP,
        batch_size=BATCH_SIZE, zoom_level=ZOOM_LEVEL, n_jobs=args.n_jobs, bg_exclude=True,
        crop_fg=True, device=device)

    test_files = read_test_files()
    print("Total of {} file(s) to process.".format(len(test_files)))
    y = np.zeros([len(test_files)], dtype=np.int)
    timing = TimingContextManager()
    total_time = 0
    for i, filename in enumerate(test_files):
        slide_path = os.path.join("data", filename)
        print("--- {} ---".format(slide_path))
        try:
            with torch.no_grad():
                with timing:
                    encoded = encoder.encode(slide_path)
                    tensor = torch.Tensor(encoded).unsqueeze(0).to(device)
                    pred = slide_classifier.forward(tensor).detach().cpu().numpy().squeeze()
                    y[i] = np.argmax(pred)
                total_time += timing.duration
                avg_time = total_time / (i + 1)
                print("> predicting              : {} / {}".format(y[i], list(pred)))
                print("> pred time               : {}s".format(timing.duration))
                print("> current average timer   : {}s".format(avg_time))
                print("> expected remaining time : {}s".format((len(test_files) - i - 1) * avg_time))
        except Exception as e:
            print("/!\\ error during prediction '{}'".format(str(e)))
            print("/!\\ ... predicting 1")
            y[i] = 1  # to minimize the error on average

    write_submission([(f, y[i]) for i, f in enumerate(test_files)])


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
