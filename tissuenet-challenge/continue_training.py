import math
import os
from argparse import ArgumentParser
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from skimage.color import rgb2hed, hed2rgb
from skimage.filters import gaussian
from skimage.util import random_noise
from sklearn.utils import check_random_state

from assets.mtdp import build_model
from assets.mtdp.networks import SingleHead
from assets.mtdp.components import Head
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from functools import partial


from svm_classifier_train import Rescale, group_per_slide, PathDataset, compute_challenge_score, compute_slide_pred


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def random_elastic_transform(img, alpha_low=80, alpha_high=120, sigma_low=9.0, sigma_high=11.0, random_state=None):
    rstate = check_random_state(random_state)
    return elastic_transform(
        img,
        alpha=rstate.randint(alpha_low, alpha_high),
        sigma=rstate.uniform(sigma_low, sigma_high),
        random_state=rstate
    )


def random_blur(img, sigma_extent=0.1, random_state=None):
    rstate = check_random_state(random_state)
    return gaussian(img, sigma=rstate.uniform(0, sigma_extent), multichannel=True)


def random_gaussian_noise(img, var_extent=0.1, random_state=None):
    rstate = check_random_state(random_state)
    return random_noise(
        img, mode='gaussian',
        var=rstate.uniform(0, var_extent) ** 2,
        seed=rstate.randint(999999))


def random_hed_ratio(img, bias_range=0.025, coef_range=0.025, random_state=None):
    rstate = check_random_state(random_state)
    hed = rgb2hed(img)
    bias = rstate.uniform(-bias_range, bias_range, 3)
    coefs = rstate.uniform(1 - coef_range, 1 + coef_range, 3)
    return np.clip(hed2rgb(hed * coefs + bias), 0, 1)


class ToNumpy(torch.nn.Module):
    def forward(self, img):
        return np.asarray(img)


class ToPillow(torch.nn.Module):
    def forward(self, img):
        return Image.fromarray((img * 255).astype(np.uint8))


def main(argv):
    """
    corrupted image: C12_B362_S12_0-1.tif
    """
    parser = ArgumentParser()
    parser.add_argument("--start_model", dest="start_model")
    parser.add_argument("-i", "--image_path", dest="image_path", default=".")
    parser.add_argument("-m", "--metadata_path", dest="metadata_path", default=".")
    parser.add_argument("-s", "--model_path", dest="model_path", default=".")
    parser.add_argument("-p", "--pretrained", dest="pretrained", default="imagenet")
    parser.add_argument("-a", "--architecture", dest="architecture", default="densenet121")
    parser.add_argument("-e", "--epochs", dest="epochs", default=5, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int)
    parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.8, type=float)
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-g", "--sched_gamma", dest="sched_gamma", default=0.1, type=float)
    parser.add_argument("-l", "--learning_rate", dest="lr", default=0.001, type=float)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
    parser.add_argument("--aug_elastic_alpha_low", dest="aug_elastic_alpha_low", type=int, default=80)
    parser.add_argument("--aug_elastic_alpha_high", dest="aug_elastic_alpha_high", type=int, default=120)
    parser.add_argument("--aug_elastic_sigma_low", dest="aug_elastic_sigma_low", type=float, default=9.0)
    parser.add_argument("--aug_elastic_sigma_high", dest="aug_elastic_sigma_high", type=float, default=11.0)
    parser.add_argument("--aug_hed_bias_range", dest="aug_hed_bias_range", type=float, default=0.025)
    parser.add_argument("--aug_hed_coef_range", dest="aug_hed_coef_range", type=float, default=0.025)
    parser.add_argument("--aug_blur_sigma_extent", dest="aug_blur_sigma_extent", type=float, default=0.1)
    parser.add_argument("--aug_noise_var_extent", dest="aug_noise_var_extent", type=float, default=0.1)
    args, _ = parser.parse_known_args(argv)

    print(args)

    start_model_filename = args.start_model
    start_epoch = int(os.path.basename(start_model_filename).split("_")[3])

    device = torch.device(args.device)
    state_dict = torch.load(args.start_model, map_location=device)
    features = build_model(arch=args.architecture, pretrained=args.pretrained, pool=True)
    model = SingleHead(features, Head(features.n_features(), n_classes=4))
    model.load_state_dict(state_dict)
    model.to(device)

    # prepare dataset
    slidenames, slide2annots, slide2cls = group_per_slide(args.metadata_path)

    post_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ]

    aug_rstate = np.random.RandomState(args.random_seed)
    train_transform = transforms.Compose([
        Rescale(args.zoom_level),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        ToNumpy(),
        partial(
            random_elastic_transform,
            alpha_low=args.aug_elastic_alpha_low,
            alpha_high=args.aug_elastic_alpha_high,
            sigma_low=args.aug_elastic_sigma_low,
            sigma_high=args.aug_elastic_sigma_high,
            random_state=aug_rstate),
        partial(
            random_gaussian_noise,
            var_extent=args.aug_noise_var_extent,
            random_state=aug_rstate),
        partial(
            random_blur,
            sigma_extent=args.aug_blur_sigma_extent,
            random_state=aug_rstate),
        partial(
            random_hed_ratio,
            bias_range=args.aug_hed_bias_range,
            coef_range=args.aug_hed_coef_range,
            random_state=aug_rstate),
        transforms.Lambda(lambda img: img.astype(np.float32))
     ] + post_trans)

    # dataset
    train_data = [(os.path.join(args.image_path, path), cls) for name in slidenames for cls, path in slide2annots[name]]
    train_dataset = PathDataset(train_data, train_transform)
    train_generator = torch.Generator("cpu")
    train_generator.manual_seed(args.random_seed)
    train_sampler = RandomSampler(train_dataset, generator=train_generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_jobs, sampler=train_sampler, pin_memory=True)

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

    end_epoch = start_epoch + args.epochs
    for e in range(start_epoch, end_epoch):
        print("Start epoch {}".format(e))
        loss_queue = list()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model.forward(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_queue = [loss.detach().cpu().item()] + loss_queue[:19]
            print("> {}: {}".format(i + 1, np.mean(loss_queue)), flush=(i % 25) == 0)
            results["train_loss"].append(loss_queue[0])

        filename = "contd_{}_{}_e_{}_z{}_{}.pth".format(args.architecture, args.pretrained, e, args.zoom_level, datetime.now().timestamp())
        torch.save(model.state_dict(), os.path.join(args.model_path, filename))


if __name__ == "__main__":
    import sys
    print(main(sys.argv[1:]))