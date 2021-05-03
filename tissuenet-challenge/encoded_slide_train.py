import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler

from svm_classifier_train import group_per_slide, compute_challenge_score


class NumpyEncodedDataset(Dataset):
    def __init__(self, slides, dirname, slide_classes=None):
        self._slides = slides
        self._dirname = dirname
        self._slide_classes = slide_classes

    def __getitem__(self, item):
        slidename = self._slides[item].rsplit(".", 1)[0]
        # print(item, slidename, flush=True)
        encoded = torch.tensor(np.load(os.path.join(self._dirname, slidename + ".npy"))).float()
        if self._slide_classes is not None:
            return encoded, self._slide_classes[item]
        else:
            return encoded

    def __len__(self):
        return len(self._slides)


def all_pad(*elements):
    max_h = np.max([elem.size()[1] for elem in elements])
    max_w = np.max([elem.size()[2] for elem in elements])
    padded = list()
    for elem in elements:
        _, h, w = elem.size()
        diff_h = max_h - h
        diff_w = max_w - w
        half_h = diff_h // 2
        half_w = diff_w // 2
        padded.append(torch.nn.ConstantPad2d([half_w, diff_w - half_w, half_h, diff_h - half_h], 0.0)(elem.unsqueeze(0)))
    return torch.cat(padded, dim=0)


def pad_collate_fn(batch):
    if isinstance(batch[0], tuple):
        return all_pad(*[t[0] for t in batch]), torch.Tensor([t[1] for t in batch]).long()
    else:
        return all_pad(*[t for t in batch])


class CustomSlideClassifier(torch.nn.Module):
    def __init__(self, features_in=1024):
        super().__init__()
        self._min_input_size = 32
        self._n_downsampling = 3
        self.inlayer = torch.nn.Conv2d(features_in, 128, kernel_size=1, bias=False)
        self.layer1 = self._make_layer(128, 32, pool=True)
        self.layer2 = self._make_layer(32, 16, pool=True)
        self.layer3 = self._make_layer(16, 4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.softmax = torch.nn.Softmax(1)

    def _make_layer(self, in_planes, out_planes, pool=False, ksize=3):
        layers = list()
        layers.append(torch.nn.Conv2d(in_planes, out_planes, kernel_size=ksize, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_planes))
        layers.append(torch.nn.ReLU())
        if pool:
            layers.append(torch.nn.AvgPool2d(ksize, stride=2))
        return torch.nn.Sequential(*layers)

    def _pad_input(self, x):
        b, c, h, w = tuple(x.size())
        mult = 2 ** self._n_downsampling
        final_h = max(self._min_input_size, h + mult - (h % mult) if h % mult > 0 else h)
        final_w = max(self._min_input_size, w + mult - (w % mult) if w % mult > 0 else w)
        diff_h = final_h - h
        diff_w = final_w - w
        return torch.nn.ConstantPad2d([0, diff_w, 0, diff_h], 0.0)(x)

    def forward(self, x):
        # padding for avoiding error on pooling
        x = self._pad_input(x)
        x = self.inlayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.softmax(x).squeeze()


class TellezClassifier(torch.nn.Module):
    def __init__(self, features_in=1024):
        super().__init__()
        self._min_input_size = 32
        self._n_downsampling = 3
        self.inlayer = torch.nn.Conv2d(features_in, 128, kernel_size=1, bias=False)
        self.layer1 = self._make_layer(128, 32, pool=True)
        self.layer2 = self._make_layer(32, 16, pool=True)
        self.layer3 = self._make_layer(16, 4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.softmax = torch.nn.Softmax(1)

    def _make_layer(self, in_planes, out_planes, pool=False, ksize=3):
        layers = list()
        layers.append(torch.nn.Conv2d(in_planes, out_planes, kernel_size=ksize, bias=False))
        layers.append(torch.nn.BatchNorm2d(out_planes))
        layers.append(torch.nn.ReLU())
        if pool:
            layers.append(torch.nn.AvgPool2d(ksize, stride=2))
        return torch.nn.Sequential(*layers)

    def _pad_input(self, x):
        b, c, h, w = tuple(x.size())
        mult = 2 ** self._n_downsampling
        final_h = max(self._min_input_size, h + mult - (h % mult) if h % mult > 0 else h)
        final_w = max(self._min_input_size, w + mult - (w % mult) if w % mult > 0 else w)
        diff_h = final_h - h
        diff_w = final_w - w
        half_diff_h = diff_h // 2
        half_diff_w = diff_w // 2
        return torch.nn.ConstantPad2d([
            half_diff_w,
            diff_w - half_diff_w,
            half_diff_h,
            diff_h - half_diff_h
        ], 0.0)(x)

    def forward(self, x):
        # padding for avoiding error on pooling
        x = self._pad_input(x)
        x = self.inlayer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.softmax(x).squeeze()


def main(argv):
    parser = ArgumentParser()
    parser.add_argument("-r", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("-p", "--base_path", dest="base_path", default=".")
    parser.add_argument("-m", "--metadata_path", dest="metadata_path", default="metadata")
    parser.add_argument("-i", "--image_path", dest="image_path", default="wsi_encoded")
    parser.add_argument("-s", "--model_path", dest="model_path", default="models/encoded")
    parser.add_argument("-e", "--epochs", dest="epochs", default=30, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int)
    parser.add_argument("-t", "--train_size", dest="train_size", default=0.8, type=float)
    parser.add_argument("-l", "--learning_rate", dest="lr", default=0.001, type=float)
    parser.add_argument("-d", "--device", dest="device", default="cpu")
    parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
    args, _ = parser.parse_known_args(argv)

    print(args)

    metadata_path = os.path.join(args.base_path, args.metadata_path)
    image_path = os.path.join(args.base_path, args.image_path)
    model_path = os.path.join(args.base_path, args.model_path)
    os.makedirs(model_path, exist_ok=True)

    # random seeds
    device = torch.device(args.device)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random_state = np.random.RandomState(args.random_seed)
    train_generator = torch.Generator("cpu")
    train_generator.manual_seed(args.random_seed)

    slidenames, slide2annots, slide2cls = group_per_slide(metadata_path)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - args.train_size, random_state=random_state)

    train_dataset = NumpyEncodedDataset(train_slides, dirname=image_path, slide_classes=[slide2cls[s] for s in train_slides])
    train_sampler = RandomSampler(train_dataset, replacement=False, generator=train_generator)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_jobs, collate_fn=pad_collate_fn)

    eval_dataset = NumpyEncodedDataset(test_slides, dirname=image_path, slide_classes=[slide2cls[s] for s in test_slides])
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=args.n_jobs, collate_fn=pad_collate_fn)

    model = CustomSlideClassifier()
    model.to(device)

    print("number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weight = torch.tensor([1.0, 1.2, 1.4, 1.6]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    losses = list()
    val_scores = list()
    models = list()
    for e in range(args.epochs):
        model.train()
        loss_queue = list()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out = model.forward(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_queue = [loss.detach().cpu().item()] + loss_queue[:19]
            print("> {}: {}".format(i + 1, np.mean(loss_queue)), flush=(i % 25) == 0)
            losses.append(loss_queue[0])

        print("start eval {}".format(e))
        model.eval()
        with torch.no_grad():
            p_pred = np.zeros([len(eval_dataset), 4], dtype=np.float)
            y_true = np.zeros([len(eval_dataset)], dtype=np.int)
            for i, (x, y) in enumerate(eval_loader):
                x = x.to(device)
                p_pred[i] = model.forward(x).detach().cpu().numpy().squeeze()
                y_true[i] = y.cpu().numpy()

            y_pred = np.argmax(p_pred, axis=1)
            val_score = compute_challenge_score(y_true, y_pred)
            print("epoch {}: {:0.6f}".format(e, val_score))
            val_scores.append(val_score)

            filename = "{}_e_{}_lr_{}_val_{:0.4f}_{}.pth".format(args.image_path.replace("/", "-"), e, args.lr, val_score, datetime.now().isoformat())
            torch.save(model.state_dict(), os.path.join(model_path, filename))

            models.append(filename)

    return {
        "train_loss": losses,
        "val_score": val_scores,
        "models": models
    }


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])