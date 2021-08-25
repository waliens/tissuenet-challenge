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
    cube = build_datacube("monuseg-unet-weights")
    param_names = [
        "no_distillation", "weights_mode", "weights_constant", "weights_consistency_fn", "weights_minimum",
        "weights_neighbourhood", "sparse_start_after", "monu_rr"
    ]
    data = list()
    headers = ["best_dice", "best_dice_std", "last_dice", "last_dice_std",
               "best_roc", "best_roc_std", "last_roc", "last_roc_std"] + param_names
    for param_values, p_cube in cube.iter_dimensions(*param_names):
        if p_cube.diagnose()["Missing ratio"] > 0.0:
            continue
        val_dice = np.array(get_metric("val_dice", p_cube))
        val_roc = np.array(get_metric("val_metrics", p_cube))
        best_dice = np.mean(np.max(val_dice, axis=1))
        best_dice_std = np.std(np.max(val_dice, axis=1))
        best_roc = np.mean(np.max(val_roc, axis=1))
        best_roc_std = np.std(np.max(val_roc, axis=1))
        last_dice = np.mean(val_dice[:, -1])
        last_dice_std = np.std(val_dice[:, -1])
        last_roc = np.mean(val_roc[:, -1])
        last_roc_std = np.std(val_roc[:, -1])

        data.append([
            "{:.4f}".format(best_dice),
            "{:.2E}".format(best_dice_std),
            "{:.4f}".format(last_dice),
            "{:.2E}".format(last_dice_std),
            "{:.4f}".format(best_roc),
            "{:.2E}".format(best_roc_std),
            "{:.4f}".format(last_roc),
            "{:.2E}".format(last_roc_std),
        ] + list(param_values))

    print("\t".join(headers))
    for d in data:
        print("\t".join(d))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
