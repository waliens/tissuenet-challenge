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
from svm_classifier_train import group_per_slide, compute_challenge_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


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
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.7, type=float)
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-i", "--image_path", dest="image_path", default=".")
    parser.add_argument("-m", "--metadata_path", dest="metadata_path", default=".")
    args, _ = parser.parse_known_args(argv)

    slidenames, slide2annots, slide2cls = group_per_slide(args.metadata_path)

    random_state = np.random.RandomState(args.random_seed)
    _, test_slides = train_test_split(slidenames, test_size=1 - args.train_size, random_state=random_state)

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

    y_pred, y_true = list(), list()
    print("{} slide(s) to process".format(len(test_slides)))
    for i, filename in enumerate(test_slides):
        with torch.no_grad():
            pred = classify(
                slide_path=os.path.join(args.image_path, filename),
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
            print("{} - {:3.2f}%".format(filename, 100 * (i + 1) / test_slides))
            y_pred.append(pred)
            y_true.append(slide2cls[filename])

    print()
    print("slide: ")
    val_slide_acc = accuracy_score(y_true, y_pred)
    val_slide_score = compute_challenge_score(y_true, y_pred)
    val_slide_cm = confusion_matrix(y_true, y_pred)
    print("> slide acc: ", val_slide_acc)
    print("> slide sco: ", val_slide_score)
    print("> slide cm : ")
    print(val_slide_cm)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
