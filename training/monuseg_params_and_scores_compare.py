import numpy as np
from clustertools import build_datacube
from matplotlib import pyplot as plt


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


def plt_with_std(ax, x, mean, std, label, color):
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.6)


def plot_current_setup(cube, axes, label, color):
    val_dice = np.array(get_metric("val_dice", cube))
    val_roc = np.array(get_metric("val_metrics", cube))

    dice_mean = np.mean(val_dice, axis=0)
    dice_std = np.std(val_dice, axis=0)
    x = np.arange(dice_mean.shape[0])

    roc_mean = np.mean(val_roc, axis=0)
    roc_std = np.std(val_roc, axis=0)

    plt_with_std(axes[0], x, dice_mean, dice_std, label, color)
    plt_with_std(axes[1], x, roc_mean, roc_std, label, color)

    return dice_mean, dice_std, roc_mean, roc_std


def make_label(names, params):
    return ", ".join(["{}={}".format(n, p) for n, p in zip(names, params)])


def main(argv):
    baseline_cube = build_datacube("monuseg-unet-baseline")

    baseline_dice = baseline_cube("val_dice")
    baseline_roc = baseline_cube("val_metrics")

    bl_dice_avg, bl_dice_std = np.mean(baseline_dice, axis=0).squeeze(), np.std(baseline_dice, axis=0).squeeze()
    bl_roc_avg, bl_roc_std = np.mean(baseline_roc, axis=0).squeeze(), np.std(baseline_roc, axis=0).squeeze()

    cube = build_datacube("monuseg-unet-missing")
    param_names = ["monu_nc", "sparse_start_after"]  # "weights_mode", "weights_constant", "weights_consistency_fn", "weights_minimum", "weights_neighbourhood",

    data = list()
    headers = ["best_dice", "best_dice_std", "last_dice", "last_dice_std",
               "best_roc", "best_roc_std", "last_roc", "last_roc_std"] + param_names

    n_ratio = len(cube.domain["monu_rr"])
    rr_map = {v: i for i, v in enumerate(sorted(cube.domain["monu_rr"], key=lambda v: float(v)))}
    for ext_param_values, ext_cube in cube.iter_dimensions(*param_names):
        if ext_cube.diagnose()["Missing ratio"] >= 1.0:
            continue

        fig, axes = plt.subplots(2, n_ratio, sharex=True, figsize=[12.8, 4.8])

        for i in range(n_ratio):
            plt_with_std(axes[0][i], np.arange(50), bl_dice_avg, bl_dice_std, label="baseline", color=COLORS[0])
            plt_with_std(axes[1][i], np.arange(50), bl_roc_avg, bl_roc_std, label="baseline", color=COLORS[0])

        dice_ymin, dice_ymax = max(np.min(bl_dice_avg) - 0.1, 0), min(np.max(bl_dice_avg) + 0.1, 1.0)
        roc_ymin, roc_ymax = max(np.min(bl_roc_avg) - 0.1, 0), min(np.max(bl_roc_avg) + 0.1, 1.0)
        color_id = 1

        for (rr, nd), in_cube in ext_cube.iter_dimensions("monu_rr", "no_distillation"):
            if in_cube.diagnose()["Missing ratio"] > 0.0:
                continue
            label = make_label(["d"], [int(not eval(nd))])

            dice_mean, dice_std, roc_mean, roc_std = plot_current_setup(
                in_cube, [axes[0][rr_map[rr]], axes[1][rr_map[rr]]], label, COLORS[color_id])

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
            axes[1][i].legend()
        plt.xlim(0, 50)

        axes[0][0].set_ylabel("val dice (opt)")
        axes[1][0].set_ylabel("val roc auc")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig("bl_" + "_".join(ext_param_values) + ".png")
        plt.close()
        print("plotted")


    print("\t".join(headers))
    for d in data:
        print("\t".join(d))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
