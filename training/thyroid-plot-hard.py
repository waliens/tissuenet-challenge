from collections import defaultdict
from pprint import pprint

import numpy as np
from clustertools import build_datacube
from matplotlib import pyplot as plt

from plot_helpers import ColorByCounter, make_label, get_metric_without_none, plt_with_std, COLORS


def get_x(cube):
    iter_per_epoch = int(cube.metadata['iter_per_epoch'])
    epochs = int(cube.metadata['epochs'])
    return np.arange(0, epochs * iter_per_epoch, iter_per_epoch)


def main():
    hard_cube = build_datacube("thyroid-unet-hard")

    to_plot = [
        "no_distillation", "weights_mode",
        "weights_consistency_fn", "weights_minimum",
        "weights_neighbourhood", "distil_target_mode"
    ]

    out_params = ["sparse_start_after", "n_calibration"]

    param_values = set()

    for _, out_cube in hard_cube.iter_dimensions(*out_params):
        for values, in_cube in out_cube.iter_dimensions(*to_plot):
            param_values.add(values)

    color_map = defaultdict(ColorByCounter(start=1))

    for (ssa, n_calib), out_cube in hard_cube.iter_dimensions(*out_params):
        plt.figure(figsize=[12.8, 4.8])
        for_params = {
            "sparse_start_after": str(ssa),
            "n_calibration": n_calib,
        }
        max_x = 0
        dice_ymin, dice_ymax = 1, 0

        print(ssa, n_calib)

        at_least_one = False

        for values, in_cube in out_cube.iter_dimensions(*to_plot):
            nd, wm, wfn, wmin, wneigh, tmode = values
            if in_cube.diagnose()["Missing ratio"] >= 1.0:
                continue
            at_least_one = True
            label = make_label(wm, {
                "distillation": int(not eval(nd)), "weights_consistency_fn": wfn,
                "weights_minimum": wmin, "weights_neighbourhood": wneigh,
                "distil_target_mode": tmode
            })

            print("> ", label)
            val_dice = np.array(get_metric_without_none(in_cube, "val_dice"))
            dice_mean = np.mean(val_dice, axis=0)
            dice_std = np.std(val_dice, axis=0)
            plt_with_std(plt.gca(), np.arange(dice_mean.shape[0]), dice_mean, dice_std, label, **color_map[values],
                         do_std=False, alpha=0.2)

            max_x = max(max_x, dice_mean.shape[0])
            dice_ymin = min(dice_ymin, np.min(dice_mean))
            dice_ymax = max(dice_ymax, np.max(dice_mean))

        title = "_".join(map(lambda t: "{}={}".format(t[0], t[1]), for_params.items()))
        plt.title(title)

        plt.ylim(dice_ymin * 0.95, dice_ymax * 1.05)
        plt.xlim(0, max_x)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylabel("val dice (opt)")
        plt.xlabel("epoch")
        plt.tight_layout()

        filename = "hard_" + title + ".pdf"

        if at_least_one:
            plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    main()