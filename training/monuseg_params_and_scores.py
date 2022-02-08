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


def main(argv):
    cube = build_datacube("monuseg-unet-missing")
    param_names = ["monu_nc", "no_distillation", "weights_mode", "weights_constant", "weights_consistency_fn", "weights_minimum", "weights_neighbourhood", "sparse_start_after"]
    param_to_plot = ["monu_rr"]

    data = list()
    headers = ["best_dice", "best_dice_std", "last_dice", "last_dice_std",
               "best_roc", "best_roc_std", "last_roc", "last_roc_std"] + param_names
    for ext_param_values, ext_cube in cube.iter_dimensions(*param_names):
        if ext_cube.diagnose()["Missing ratio"] >= 1.0:
            continue

        fig, axs = plt.subplots(2, 1, sharex=True)
        ax1, ax2 = axs

        dice_ymin, dice_ymax = 1, 0
        roc_ymin, roc_ymax = 1, 0
        for i, (in_param_values, in_cube) in enumerate(ext_cube.iter_dimensions(*param_to_plot)):
            if in_cube.diagnose()["Missing ratio"] > 0.0:
                continue

            val_dice = np.array(get_metric("val_dice", in_cube))
            val_roc = np.array(get_metric("val_metrics", in_cube))

            dice_mean = np.mean(val_dice, axis=0)
            dice_std = np.std(val_dice, axis=0)
            x = np.arange(dice_mean.shape[0])

            roc_mean = np.mean(val_roc, axis=0)
            roc_std = np.std(val_roc, axis=0)

            ax1.plot(x, dice_mean, label=in_param_values[0], color=COLORS[i])
            ax1.fill_between(x, dice_mean - dice_std, dice_mean + dice_std,
                             color=COLORS[i], alpha=0.6)

            ax2.plot(x, roc_mean, label=in_param_values[0], color=COLORS[i])
            ax2.fill_between(x, roc_mean - roc_std, roc_mean + roc_std,
                             color=COLORS[i], alpha=0.6)

            dice_ymin = min(dice_ymin, np.min(dice_mean))
            dice_ymax = max(dice_ymax, np.max(dice_mean))
            roc_ymin = min(roc_ymin, np.min(roc_mean))
            roc_ymax = max(roc_ymax, np.max(roc_mean))

        plt.title("_".join(ext_param_values))
        ax1.set_xlim(0, 50)
        ax1.set_ylim(max(dice_ymin - 0.1, 0), min(1, dice_ymax + 0.1))
        ax2.set_ylim(max(roc_ymin - 0.1, 0), min(1, roc_ymax + 0.1))
        ax2.legend()
        ax1.set_ylabel("val dice (opt)")
        ax2.set_ylabel("val roc auc")
        ax2.set_xlabel("epoch")
        plt.savefig("_".join(ext_param_values) + ".png")
        plt.close()

            # best_dice = np.mean(np.max(val_dice, axis=1))
            # best_dice_std = np.std(np.max(val_dice, axis=1))
            # best_roc = np.mean(np.max(val_roc, axis=1))
            # best_roc_std = np.std(np.max(val_roc, axis=1))
            # last_dice = np.mean(val_dice[:, -1])
            # last_dice_std = np.std(val_dice[:, -1])
            # last_roc = np.mean(val_roc[:, -1])
            # last_roc_std = np.std(val_roc[:, -1])
            #
            # data.append([
            #     "{:.4f}".format(best_dice),
            #     "{:.2E}".format(best_dice_std),
            #     "{:.4f}".format(last_dice),
            #     "{:.2E}".format(last_dice_std),
            #     "{:.4f}".format(best_roc),
            #     "{:.2E}".format(best_roc_std),
            #     "{:.4f}".format(last_roc),
            #     "{:.2E}".format(last_roc_std),
            # ] + list(param_values))

    print("\t".join(headers))
    for d in data:
        print("\t".join(d))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
