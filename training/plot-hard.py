from pprint import pprint

import numpy as np
from monuseg_params_and_scores_compare import get_metric, plt_with_std, COLORS
from clustertools import build_datacube
from matplotlib import pyplot as plt


baseline_cube = build_datacube("monuseg-unet-baseline")
baseline_dice = baseline_cube("val_dice")
bl_dice_avg, bl_dice_std = np.mean(baseline_dice, axis=0).squeeze(), np.std(baseline_dice, axis=0).squeeze()
del baseline_cube
cube = build_datacube("monuseg-unet-hard")


to_plot = [
    "monu_rr", "no_distillation", "weights_mode",
    "weights_consistency_fn", "weights_minimum",
    "weights_neighbourhood", "distil_target_mode"
]

out_params = ["monu_rr", "monu_nc", "sparse_start_after", "n_calibration"]

param_values = set()

for _, out_cube in cube.iter_dimensions(*out_params):
    for values, in_cube in out_cube.iter_dimensions(*to_plot):
        param_values.add(values)

param_val_idxs = {v: i for i, v in enumerate(sorted(list(param_values)))}



def readable_weights_mode(wm):
    return {
        "pred_entropy": "entr",
        "pred_merged": "merg",
        "constant": "csnt",
        "balance_gt": "bala",
        "pred_consistency": "csty"
    }.get(wm, "n/a")


def make_label(wmode, params):
    n, v = ["w", "d", "m"], [readable_weights_mode(wmode), params['distillation'], params["distil_target_mode"]]
    if wmode == "pred_consistency" or wmode == "pred_merged":
        n.extend(["nh", "fn"])
        v.extend([params["weights_neighbourhood"], params["weights_consistency_fn"][:4]])
    elif not (wmode == "constant" or wmode == "balance_gt" or wmode == 'pred_entropy'):
        raise ValueError("unknown wmode '{}'".format(wmode))
    if wmode != "constant":
        n.append("wmin")
        v.append(params['weights_minimum'])
    return ", ".join(["{}={}".format(n, p) for n, p in zip(n, v)])


for (monu_rr, monu_nc, ssa, n_calib), out_cube in cube.iter_dimensions(*out_params):
    plt.figure(figsize=[12.8, 4.8])
    for_params = {
        "monu_rr": str(monu_rr),
        "monu_nc": str(monu_nc),
        "sparse_start_after": str(ssa),
        "n_calibration": n_calib,
    }

    plt_with_std(plt.gca(), np.arange(50), bl_dice_avg, bl_dice_std, label="baseline", color=COLORS[0])

    dice_ymin, dice_ymax = np.min(bl_dice_avg), np.max(bl_dice_avg)

    print(monu_rr, monu_nc, ssa, n_calib)

    at_least_one = False

    for values, in_cube in out_cube.iter_dimensions(*to_plot):
        rr, nd, wm, wfn, wmin, wneigh, tmode = values
        if wm == "pred_merged" or wm == "pred_consistency":
            continue
        if in_cube.metadata["monu_nc"] == "4" and 0.49 < float(in_cube.metadata["monu_rr"]) < 0.51 and in_cube.metadata["n_calibration"] == "1":
            if in_cube.diagnose()["Missing ratio"] < 1.0:
                print("missing ? ", in_cube.metadata, in_cube.diagnose())
        if in_cube.diagnose()["Missing ratio"] > 0.0:
            continue
        at_least_one = True
        label = make_label(wm, {
            "monu_rr": rr, "distillation": int(not eval(nd)),
            "weights_consistency_fn": wfn,
            "weights_minimum": wmin, "weights_neighbourhood": wneigh,
            "distil_target_mode": tmode
        })

        print("> ", label)
        val_dice = np.array(get_metric("val_dice", in_cube))
        dice_mean = np.mean(val_dice, axis=0)
        dice_std = np.std(val_dice, axis=0)
        x = np.arange(dice_mean.shape[0])

        plt_with_std(plt.gca(), x, dice_mean, dice_std, label, COLORS[(param_val_idxs[values] + 1) % len(COLORS)],
                     do_std=False, alpha=0.2)

        dice_ymin = min(dice_ymin, np.min(dice_mean))
        dice_ymax = max(dice_ymax, np.max(dice_mean))

    title = "_".join(map(lambda t: "{}={}".format(t[0], t[1]), for_params.items()))
    plt.title(title)

    plt.ylim(dice_ymin * 0.95, dice_ymax * 1.05)
    plt.xlim(0, 50)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel("val dice (opt)")
    plt.xlabel("epoch")
    plt.tight_layout()

    filename = "hard_" + title + ".pdf"

    if at_least_one:
        plt.savefig(filename)
    plt.close()


