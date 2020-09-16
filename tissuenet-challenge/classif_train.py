from argparse import ArgumentParser

import torch
import numpy as np
from PIL.Image import Image
from mtdp import build_model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def rescale_fn(zoom, to):
    def rescale(im):
        factor = 2 ** zoom
        w, h = im.width // factor, im.height // factor
        wd, hd = w % 64, h % 64
        out_im = im.resize((h + (64 - hd), w + (64 - wd))).resize((to, to))
        return out_im
    return rescale


def main(argv):
    """
    corrupted image: C12_B362_S12_0-1.tif
    """
    parser = ArgumentParser()
    parser.add_argument("-p", "--pretrained", dest="pretrained", default="imagenet")
    parser.add_argument("-a", "--architecture", dest="architecture", default="densenet121")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int)
    parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.7, type=float)
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-s", "--out_size", dest="out_size", default=384, type=int)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
    args, _ = parser.parse_known_args(argv)

    device = torch.device(args.device)
    model = build_model(arch=args.architecture, pretrained=args.pretrained, pool=True)
    model.to(device)

    transform = transforms.Compose([
        transforms.Lambda(rescale_fn(args.zoom_level, args.out_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    dataset = ImageFolder("D:/tmp", transform=transform)
    n_samples = len(dataset)
    n_samples_train = int(n_samples * args.train_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.random_seed)
    d_train, d_val = random_split(dataset, [n_samples_train, n_samples - n_samples_train], generator=generator)

    train_loader = DataLoader(d_train, batch_size=args.batch_size)
    val_loader = DataLoader(d_val, batch_size=args.batch_size)
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
        for i, (x_test, y) in enumerate(val_loader):
            print("> test iter #{}".format(i + 1))
            out = model.forward(x_test.to(device))
            preds.append(svm.predict(out.detach().cpu().numpy().squeeze()))
            y_test.append(y.cpu().numpy())

        preds = np.hstack(preds)
        y_test = np.hstack(y_test)

        print("test accuracy:", accuracy_score(y_test, preds))

        error_matrix = np.array([
            [0.0, 0.1, 0.7, 1.0],
            [0.1, 0.0, 0.3, 0.7],
            [0.7, 0.3, 0.0, 0.3],
            [1.0, 0.7, 0.3, 0.0]
        ])
        cm = np.array(confusion_matrix(y_test, preds))
        print("confusion matrix")
        print(np.round(100 * cm / np.sum(cm), 2))
        print("error cm")
        error_cm = cm * error_matrix
        print(error_cm)
        print("error: ", np.sum(error_cm) / np.sum(error_matrix))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])