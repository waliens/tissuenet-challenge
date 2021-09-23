import os
from collections import defaultdict

import imageio
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def split_filename(f):
    epoch, rest = f.split("_", 1)
    identifier, _type = rest.rsplit(".", 1)[0].rsplit("_", 1)
    return epoch, identifier, _type


def get_concat_h(im1, im2, mode=None):
    dst = Image.new(im1.mode if mode is None else mode, (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2, mode=None):
    dst = Image.new(im1.mode if mode is None else mode, (max(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def _find_by_walk(query, dir):
    for dirpath, dirnames, filenames in os.walk(dir, topdown=False):
        for filename in filenames:
            if filename == query:
                return os.path.join(dirpath, filename)
    return None


def colorize(gt, pred, border=False):
    np_gt = np.asarray(gt)
    np_pred = np.asarray(pred)
    colored = np.zeros([np_gt.shape[0], np_gt.shape[1], 3], dtype=np.uint8)
    fp_mask = np.logical_and(np_gt > 0, np_gt != np_pred)
    fn_mask = np.logical_and(np_gt > 0, np_gt == np_pred)
    tp_mask = np.logical_and(np_gt == 0, np_gt != np_pred)
    colored[fp_mask, 0] = 255
    colored[fn_mask, 1] = 255
    colored[tp_mask, 0] = 255
    colored[tp_mask, 1] = 179
    colored[tp_mask, 2] = 25
    if border:
        struct = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=np.bool)
        er_fp_mask = binary_erosion(fp_mask, struct)
        er_fn_mask = binary_erosion(fn_mask, struct)
        er_tp_mask = binary_erosion(tp_mask, struct)
        colored[er_fp_mask, 0] = 0
        colored[er_fn_mask, 1] = 0
        colored[er_tp_mask, 0] = 0
        colored[er_tp_mask, 1] = 0
        colored[er_tp_mask, 2] = 0
        mask = np.zeros(colored.shape[:2] + (1,), dtype=np.uint8)
        mask[np.logical_or(np_gt > 0, np_pred > 0)] = 255
        return Image.fromarray(np.concatenate([colored, mask], axis=-1))
    else:
        return Image.fromarray(colored)


def main(argv):
    directory = "/home/rmormont/images/monuseg"

    filenames = os.listdir(directory)
    thre_group_by_epoch_dict = defaultdict(list)
    prob_group_by_epoch_dict = defaultdict(list)
    for f in filenames:
        if f.endswith(".gif"):
            continue
        epoch, identifier, _type = split_filename(f)
        if _type == "prob":
            prob_group_by_epoch_dict[identifier].append((int(epoch), f))
        elif _type == "thre":
            thre_group_by_epoch_dict[identifier].append((int(epoch), f))
        else:
            raise ValueError("error")

    for identifier in thre_group_by_epoch_dict.keys():
        print(identifier)
        prob = sorted(prob_group_by_epoch_dict[identifier], key=lambda l: l[0])
        thre = sorted(thre_group_by_epoch_dict[identifier], key=lambda l: l[0])

        filename = identifier.rsplit("_", 1)[1]
        _in_fname = _find_by_walk(filename + ".tif", "/scratch/users/rmormont/monuseg/42_0.0000_30/")
        _in = Image.open(_in_fname)
        _in.putalpha(255)
        _gt_fname = _find_by_walk(filename + ".png", "/scratch/users/rmormont/monuseg/42_0.0000_30/")
        _gt = Image.open(_gt_fname)
        with imageio.get_writer(os.path.join(directory, "{}.gif".format(identifier)), mode="I", duration=0.5) as writer:
            for (ep, prob_f), (et, thre_f) in zip(prob, thre):
                if ep != et:
                    raise ValueError("Error")
                image_prob = Image.open(os.path.join(directory, prob_f))
                image_thre = Image.open(os.path.join(directory, thre_f))
                only_thre = image_thre.crop((0, 0, 1000, 1000))
                image = get_concat_v(
                    get_concat_h(image_prob, image_thre, mode="RGBA"),
                    get_concat_h(Image.alpha_composite(_in, colorize(_gt, only_thre, border=True)),
                                 colorize(_gt, only_thre)))
                image = image.resize((image.width // 2, image.height // 2))
                writer.append_data(np.asarray(image))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
