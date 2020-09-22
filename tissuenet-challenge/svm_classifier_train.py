import csv
import os
from argparse import ArgumentParser
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from mtdp import build_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PathDataset(Dataset):
    def __init__(self, samples, trans):
        self._files, self._classes = zip(*samples)
        self._trans = trans

    def __getitem__(self, item):
        file = self._files[item]
        image = Image.open(file)
        return self._trans(image), self._classes[item]

    def __len__(self):
        return len(self._files)


class Rescale(object):
    def __init__(self, zoom):
        self._zoom = zoom

    def rescale(self, im):
        norm_im = im.resize((1280, 1280))
        factor = 2 ** self._zoom
        w, h = norm_im.width // factor, norm_im.height // factor
        out_im = norm_im.resize((h, w))
        return out_im

    def __call__(self, im):
        return self.rescale(im)


def group_per_slide(path):
    """
    :return
    1) list of slide filenames
    2) map {slide filename} => [(cls1, path1), (cls2, path2), ...]
    3) map {slide filename} => {slide cls}
    """
    with open(os.path.join(path, "train_annotations_lbzOVuS.csv")) as annot_file:
        annot_reader = csv.DictReader(annot_file,
                                      fieldnames=["annotation_id", "filename", "geometry", "annotation_class",
                                                  "us_jpeg_url", "eu_jpeg_url", "asia_jpeg_url"])
        next(annot_reader)
        file2annots = defaultdict(list)
        for row in annot_reader:
            _, ext = row['filename'].rsplit(".", 1)
            filepath = os.path.join(row['annotation_class'], row['annotation_id'] + "." + ext)
            cls = int(row['annotation_class'])
            file2annots[row["filename"]].append((cls, filepath))
        return list(file2annots.keys()), file2annots, {f: max([cls for cls, _ in v]) for f, v in file2annots.items()}


def compute_slide_score(slide2cls, paths, preds):
    """
    :param slides: dict
    :param paths: (n,)
    :param preds: (n,)
    :return:
    """
    slide2pred = defaultdict(lambda: -1)
    for path, pred in zip(paths, preds):
        slide_name = os.path.basename(path).rsplit("_", 1)[0]
        slide2pred[slide_name] = max(slide2pred[slide_name], pred)
    y_pred, y_true = list(), list()
    for filename, cls in slide2pred.items():
        y_pred.append(cls)
        y_true.append(slide2cls[filename + ".tif"])
    return np.array(y_true), np.array(y_pred)


def compute_error(y_true, y_pred):
    error_matrix = np.array([
        [0.0, 0.1, 0.7, 1.0],
        [0.1, 0.0, 0.3, 0.7],
        [0.7, 0.3, 0.0, 0.3],
        [1.0, 0.7, 0.3, 0.0]
    ])
    return np.mean(1 - error_matrix[y_true, y_pred])


def main(argv):
    """
    corrupted image: C12_B362_S12_0-1.tif
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_path", dest="image_path", default=".")
    parser.add_argument("-m", "--metadata_path", dest="metadata_path", default=".")
    parser.add_argument("-p", "--pretrained", dest="pretrained", default="imagenet")
    parser.add_argument("-a", "--architecture", dest="architecture", default="densenet121")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int)
    parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.7, type=float)
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
    args, _ = parser.parse_known_args(argv)

    device = torch.device(args.device)
    model = build_model(arch=args.architecture, pretrained=args.pretrained, pool=True)
    model.to(device)

    # prepare dataset
    slidenames, slide2annots, slide2cls = group_per_slide(args.metadata_path)

    random_state = np.random.RandomState(args.random_seed)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - args.train_size, random_state=random_state)

    # transform
    transform = transforms.Compose([
        transforms.Lambda(Rescale(args.zoom_level)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # dataset
    train_data = [(os.path.join(args.image_path, path), cls) for name in train_slides for cls, path in
                  slide2annots[name]]
    test_data = [(os.path.join(args.image_path, path), cls) for name in test_slides for cls, path in slide2annots[name]]
    train_dataset = PathDataset(train_data, transform)
    test_dataset = PathDataset(test_data, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_jobs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_jobs)

    with torch.no_grad():
        model.eval()
        # train
        features = list()
        classes = list()
        for i, (x, y) in enumerate(train_loader):
            print("> train iter #{}".format(i + 1))
            out = model.forward(x.to(device))
            features.append(out.detach().cpu().numpy().squeeze())
            classes.append(y.cpu().numpy())

        features = np.vstack(features)
        classes = np.hstack(classes)

        print("Train svm.")
        svm = LinearSVC(C=0.01)
        svm.fit(features, classes)

        # predict
        preds = list()
        y_test = list()
        for i, (x_test, y) in enumerate(test_loader):
            print("> test iter #{}".format(i + 1))
            out = model.forward(x_test.to(device))
            preds.append(svm.predict(out.detach().cpu().numpy().squeeze()))
            y_test.append(y.cpu().numpy())

        preds = np.hstack(preds)
        y_test = np.hstack(y_test)

        print("window:")
        print("> acc: ", accuracy_score(y_test, preds))
        print("> sco: ", compute_error(y_test, preds))
        print("> cm : ")
        print(confusion_matrix(y_test, preds))
        print()
        print("slide: ")
        slide_true, slide_pred = compute_slide_score(slide2cls, list(zip(*test_data))[0], preds)
        print("> acc: ", accuracy_score(slide_true, slide_pred))
        print("> sco: ", compute_error(slide_true, slide_pred))
        print("> cm : ")
        print(confusion_matrix(slide_true, slide_pred))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])