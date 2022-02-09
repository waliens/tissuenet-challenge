import itertools
from collections import defaultdict
from pprint import pprint

import numpy as np
from clustertools import build_datacube
from matplotlib import pyplot as plt

import imageio

from itertools import chain


COLORS = [
    "#1f78b4",
    "#33a02c",
    "#e31a1c",
    "#ff7f00",
    "#6a3d9a",
    "#b15928",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f"
]


def get_color(i):
    return COLORS[i % len(COLORS)]


def get_metric(metric_name, cube):
    return [p_cube(metric_name) for p_values, p_cube in cube.iter_dimensions(*cube.parameters)]


def plt_with_std(ax, x, mean, std, label, color, do_std=True, alpha=0.6):
    ax.plot(x, mean, label=label, color=color)
    if do_std:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def plot_current_setup(cube, axes, label, color):
    val_dice = np.array(get_metric("val_dice", cube))
    val_roc = np.array(get_metric("val_metrics", cube))
    val_thresh = np.array(get_metric("threshold", cube))

    dice_mean = np.mean(val_dice, axis=0)
    dice_std = np.std(val_dice, axis=0)
    x = np.arange(dice_mean.shape[0])

    roc_mean = np.mean(val_roc, axis=0)
    roc_std = np.std(val_roc, axis=0)

    thresh_mean = np.mean(val_thresh, axis=0)
    thresh_std = np.std(val_thresh, axis=0)

    plt_with_std(axes[0], x, dice_mean, dice_std, label, color)
    plt_with_std(axes[1], x, roc_mean, roc_std, label, color)
    plt_with_std(axes[2], x, thresh_mean, thresh_std, label, color)

    return dice_mean, dice_std, roc_mean, roc_std, thresh_mean, thresh_std


def make_label(wmode, params):
    n, v = ["w", "d"], [readable_weights_mode(wmode), params['distillation']]
    if wmode == "pred_consistency" or wmode == "pred_merged":
        n.extend(["nh", "fn"])
        v.extend([params["weights_neighbourhood"], params["weights_consistency_fn"][:4]])
    elif not (wmode == "constant" or wmode == 'pred_entropy'):
        raise ValueError("unknown wmode '{}'".format(wmode))
    if wmode != "constant":
        n.append("wmin")
        v.append(params['weights_minimum'])
    return ", ".join(["{}={}".format(n, p) for n, p in zip(n, v)])


def readable_weights_mode(wm):
    return {
        "pred_entropy": "entr",
        "pred_merged": "merg",
        "constant": "csnt",
        "balance_gt": "bala",
        "pred_consistency": "csty"
    }.get(wm, "n/a")


def main(argv):
    baseline_cube = build_datacube("monuseg-unet-baseline")

    baseline_dice = baseline_cube("val_dice")
    baseline_roc = baseline_cube("val_metrics")

    bl_dice_avg, bl_dice_std = np.mean(baseline_dice, axis=0).squeeze(), np.std(baseline_dice, axis=0).squeeze()
    bl_roc_avg, bl_roc_std = np.mean(baseline_roc, axis=0).squeeze(), np.std(baseline_roc, axis=0).squeeze()

    del baseline_cube

    cube = build_datacube("monuseg-unet-missing")(weights_mode="constant")
    weights_cube = build_datacube("monuseg-unet-weights")(weights_consistency_fn="quadratic")

    param_names = ["monu_nc", "sparse_start_after"]  # "weights_mode", "weights_constant", "weights_consistency_fn", "weights_minimum", "weights_neighbourhood",

    # data = list()
    # headers = ["best_dice", "best_dice_std", "last_dice", "last_dice_std",
    #            "best_roc", "best_roc_std", "last_roc", "last_roc_std"] + param_names

    color_ids = {}
    next_avail_color_index = 0

    filenames_by_ssa = defaultdict(list)

    n_ratio = len(cube.domain["monu_rr"])
    rr_map = {v: i for i, v in enumerate(sorted(cube.domain["monu_rr"], key=lambda v: float(v)))}
    for ext_param_values, ext_cube in chain(cube.iter_dimensions(*param_names), weights_cube.iter_dimensions(*param_names)):
        if ext_cube.diagnose()["Missing ratio"] >= 1.0:
            continue

        fig, axes = plt.subplots(3, n_ratio, sharex=True, figsize=[12.8, 7.2])

        for i in range(n_ratio):
            plt_with_std(axes[0][i], np.arange(50), bl_dice_avg, bl_dice_std, label="baseline", color=COLORS[0])
            plt_with_std(axes[1][i], np.arange(50), bl_roc_avg, bl_roc_std, label="baseline", color=COLORS[0])

        dice_ymin, dice_ymax = max(np.min(bl_dice_avg) - 0.1, 0), min(np.max(bl_dice_avg) + 0.1, 1.0)
        roc_ymin, roc_ymax = max(np.min(bl_roc_avg) - 0.1, 0), min(np.max(bl_roc_avg) + 0.1, 1.0)
        color_id = 1

        for (rr, nd, wm, wfn, win, wneigh), in_cube in ext_cube.iter_dimensions(
            "monu_rr", "no_distillation", "weights_mode",
            "weights_consistency_fn", "weights_minimum", "weights_neighbourhood"
        ):
            if in_cube.diagnose()["Missing ratio"] > 0.0:
                continue
            color_key = tuple(map(str, (rr, nd, wm, wfn, win, wneigh)))
            color = color_ids.get(color_key)
            if color is None:
                color_ids[color_key] = next_avail_color_index
                color = next_avail_color_index
                next_avail_color_index += 1

            label = make_label(wm, {
                "monu_rr": rr, "distillation": int(not eval(nd)),
                "weights_consistency_fn": wfn,
                "weights_minimum": win, "weights_neighbourhood": wneigh
            })

            dice_mean, dice_std, roc_mean, roc_std, thresh_mean, thresh_std = plot_current_setup(
                in_cube, [axes[i][rr_map[rr]] for i in range(3)], label, COLORS[color % len(COLORS)])

            dice_ymin = min(dice_ymin, np.min(dice_mean))
            dice_ymax = max(dice_ymax, np.max(dice_mean))
            roc_ymin = min(roc_ymin, np.min(roc_mean))
            roc_ymax = max(roc_ymax, np.max(roc_mean))

            color_id += 1

        plt.suptitle("_".join(ext_param_values))

        for v, i in rr_map.items():
            axes[0][i].set_title("% = {}".format(v))
            axes[0][i].set_ylim(max(dice_ymin - 0.1, 0), min(1, dice_ymax + 0.1))
            axes[1][i].set_ylim(max(roc_ymin - 0.1, 0), min(1, roc_ymax + 0.1))
            axes[2][i].set_ylim(0, 1)
        axes[1][0].legend()
        plt.xlim(0, 50)

        axes[0][0].set_ylabel("val dice (opt)")
        axes[1][0].set_ylabel("val roc auc")
        axes[2][0].set_ylabel("opt threshold")
        plt.xlabel("epoch")
        plt.tight_layout()
        filename = "bl_" + "_".join(ext_param_values) + ".png"
        filenames_by_ssa[ext_param_values[1]].append((ext_param_values[0], filename))
        plt.savefig(filename)
        plt.close()

    for ssa, files in filenames_by_ssa.items():
        with imageio.get_writer("{}.gif".format(ssa), mode="I", duration=1) as writer:
            for _, filename in sorted(files, key=lambda t: int(t[0])):
                image = imageio.imread(filename)
                writer.append_data(image)



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
