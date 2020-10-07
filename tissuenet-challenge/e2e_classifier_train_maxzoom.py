import math
import os
from argparse import ArgumentParser
from datetime import datetime

import torch
import numpy as np
from assets.mtdp.networks import SingleHead
from assets.mtdp.components import Head
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from assets.networks import  MultiScaleNetwork
from svm_classifier_train import Rescale, group_per_slide, PathDataset, compute_challenge_score, compute_slide_pred


def main(argv):
    """
    corrupted image: C12_B362_S12_0-1.tif
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_path", dest="image_path", default=".")
    parser.add_argument("-m", "--metadata_path", dest="metadata_path", default=".")
    parser.add_argument("-s", "--model_path", dest="model_path", default=".")
    parser.add_argument("-e", "--epochs", dest="epochs", default=5, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int)
    parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.7, type=float)
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-l", "--learning_rate", dest="lr", default=0.001, type=float)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
    args, _ = parser.parse_known_args(argv)

    print(args)

    device = torch.device(args.device)
    features = MultiScaleNetwork(6, [0, 1, 2], n_out_blocks=3)
    model = SingleHead(features, Head(features.n_features(), n_classes=4))
    model.to(device)

    # prepare dataset
    slidenames, slide2annots, slide2cls = group_per_slide(args.metadata_path)

    random_state = np.random.RandomState(args.random_seed)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - args.train_size, random_state=random_state)

    # transform
    post_trans = [
        transforms.Lambda(Rescale(args.zoom_level)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ]

    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
     ] + post_trans)

    test_transform = transforms.Compose(post_trans)

    # dataset
    train_data = [(os.path.join(args.image_path, path), cls) for name in train_slides for cls, path in slide2annots[name]]
    test_data = [(os.path.join(args.image_path, path), cls) for name in test_slides for cls, path in slide2annots[name]]
    train_dataset = PathDataset(train_data, train_transform)
    test_dataset = PathDataset(test_data, test_transform)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.random_seed)
    train_sampler = RandomSampler(train_dataset, num_samples=len(train_dataset), replacement=True, generator=train_generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_jobs, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_jobs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weight = torch.tensor([1.0, 1.2, 1.4, 1.6]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    results = {
        "train_loss": [],
        "val_roc": [],
        "val_acc": [],
        "val_score": [],
        "val_cm": [],
        "val_slide_acc": [],
        "val_slide_score": [],
        "val_slide_cm": [],
        "n_iter_per_epoch": math.ceil(len(train_dataset) / args.batch_size),
        "models": []
    }

    for e in range(args.epochs):
        model.train()
        loss_queue = list()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_queue = [loss.detach().cpu().item()] + loss_queue[:19]
            print("> {}: {}".format(i + 1, np.mean(loss_queue)))
            results["train_loss"].append(loss_queue[0])

        with torch.no_grad():
            model.eval()
            # predict
            probas = list()
            y_test = list()
            for i, (x_test, y) in enumerate(test_loader):
                out = torch.nn.functional.softmax(model.forward(x_test.to(device)), dim=1)
                probas.append(out.detach().cpu().numpy().squeeze())
                y_test.append(y.cpu().numpy())

            probas = np.vstack(probas)
            y_test = np.hstack(y_test)
            y_pred = np.argmax(probas, axis=1)

            print("window:")
            val_acc = accuracy_score(y_test, y_pred)
            val_roc = roc_auc_score(y_test, probas, multi_class='ovo')
            val_score = compute_challenge_score(y_test, y_pred)
            val_cm = confusion_matrix(y_test, y_pred)
            print("> acc: ", val_acc)
            print("> roc: ", val_roc)
            print("> sco: ", val_score)
            print("> cm : ")
            print(val_cm)

            print()
            print("slide: ")
            slide_true, slide_pred = compute_slide_pred(slide2cls, list(zip(*test_data))[0], y_pred)
            val_slide_acc = accuracy_score(slide_true, slide_pred)
            val_slide_score = compute_challenge_score(slide_true, slide_pred)
            val_slide_cm = confusion_matrix(slide_true, slide_pred)
            print("> slide acc: ", val_slide_acc)
            print("> slide sco: ", val_slide_score)
            print("> slide cm : ")
            print(val_slide_cm)

            results["val_roc"].append(val_roc)
            results["val_acc"].append(val_acc)
            results["val_score"].append(val_score)
            results["val_cm"].append(val_cm)
            results["val_slide_acc"].append(val_slide_acc)
            results["val_slide_score"].append(val_slide_score)
            results["val_slide_cm"].append(val_slide_cm)

            filename = "e_{}_val_{:0.4f}_sco_{:0.4f}_z{}_{}.pth".format(e, val_acc, val_score, args.zoom_level, datetime.now().timestamp())
            torch.save(model.state_dict(), os.path.join(args.model_path, filename))

            results["models"].append(filename)

    return results


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1:]))