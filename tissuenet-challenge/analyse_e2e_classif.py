import os

import numpy as np
from clustertools import build_datacube
from matplotlib import pyplot as plt


def moving_average(data, alpha=0.75):
    if len(data) == 0:
        return []
    avged = [data[0]]
    for point in data[1:]:
        avged.append(avged[-1] * (1 - alpha) + point * alpha)
    return avged


if __name__ == "__main__":

    datacube = build_datacube("tissuenet-e2e-train-2nd")
    plot_path = "/scratch/users/rmormont/tissuenet/plots"

    os.makedirs(plot_path, exist_ok=True)

    best_comb = None
    best_epoch = None
    best_score = 0

    for (arch, pre, lr, zoom, bs), cube in datacube.iter_dimensions("architecture", "pretrained", "learning_rate", "zoom_level", "batch_size"):
        if cube('val_acc') is None:
            continue
        plt.figure()

        plt.title("{} {} {} {}".format(arch, pre, lr, zoom))
        train_loss = cube("train_loss")
        val_acc = cube("val_acc")
        val_slide_score = cube("val_slide_score")
        x = np.arange(0, len(train_loss))
        x_val = (np.arange(0, len(val_acc)) + 1) * cube('n_iter_per_epoch')
        plt.plot(x, moving_average(train_loss), label="train_loss")
        plt.plot(x_val, val_acc, label="val_acc")
        plt.plot(x_val, val_slide_score, label="val_slide_score")
        plt.legend()
        plt.ylim(0, 2)
        plt.savefig(os.path.join(plot_path, "{}_{}_{}_{}.png".format(arch, pre, lr, zoom)), dpi=300)
        plt.close()

        if np.max(val_acc) > best_score:
            best_comb = (arch, pre, lr, zoom)
            best_score = np.max(val_acc)
            best_epoch = np.argmax(val_acc)

    for (arch, pre, lr), cube in datacube.iter_dimensions("architecture", "pretrained", "learning_rate"):
        plt.figure()
        plt.title("{} {} {}".format(arch, pre, lr))
        min_curve = 1
        for (zoom, bs), zoom_cube in cube.iter_dimensions("zoom_level", "batch_size"):
            if zoom_cube("val_roc") is None:
                continue
            val_acc = zoom_cube("val_acc")
            x_val = np.arange(0, len(val_acc)) + 1
            plt.plot(x_val, val_acc, label=zoom + " (best: {:0.4f})".format(np.max(val_acc)))
            min_curve = min(np.min(val_acc), min_curve)
        plt.legend()
        plt.ylabel("acc")
        plt.ylim(0.9 * min_curve, 1)
        plt.savefig(os.path.join(plot_path, "zoom_{}_{}_{}.png".format(arch, pre, lr)), dpi=300)
        plt.close()

    for (lr, ), cube in datacube(zoom_level='2', batch_size='32').iter_dimensions("learning_rate"):
        plt.figure()
        plt.title("{} zoom=2".format(lr))
        min_curve = 1
        for (arch, pre), arch_cube in cube.iter_dimensions("architecture", "pretrained"):
            val_acc = arch_cube("val_acc")
            if val_acc is None:
                continue
            x_val = np.arange(0, len(val_acc)) + 1
            plt.plot(x_val, val_acc, label="{} - {} (best: {:0.4f})".format(arch, pre, np.max(val_acc)))
            min_curve = min(np.min(val_acc), min_curve)
        plt.legend()
        plt.ylabel("acc")
        plt.ylim(0.9 * min_curve, 1)
        plt.savefig(os.path.join(plot_path, "arch_z2_{}.png".format(lr)), dpi=300)
        plt.close()

    for (lr, ), lr_cube in datacube.iter_dimensions("learning_rate"):
        plt.figure()
        plt.title("lr={}".format(lr))
        for (arch, pre, zoom, bs), cube in lr_cube.iter_dimensions("architecture", "pretrained", "zoom_level", "batch_size"):
            if cube('val_acc') is None:
                continue
            val_acc = cube("val_acc")
            x_val = np.arange(0, len(val_acc))
            plt.plot(x_val, val_acc, label="a={} p={} z={}".format(arch[:3], pre[:3], zoom))
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(os.path.join(plot_path, "all_{}.png".format(lr)), dpi=300)
        plt.close()
