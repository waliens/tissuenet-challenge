from collections import defaultdict

import numpy as np
from clustertools.parameterset import build_parameter_set

from clustertools import build_datacube
from matplotlib import pyplot as plt

from plot_helpers import create_comp_index, plt_with_std, COLORS, make_label, get_metric_by_comp_index, ColorByCounter
from train_monuseg_hard_clustertools import weight_exclude, exclude_target_and_dice_calibration, no_distillation_filter, \
    filter_nc_rr, min_weight_only_for_entropy


def main():
    baseline_cube = build_datacube("monuseg-reeval-baseline")
    baseline_dice = baseline_cube("val_hard_dice")
    bl_dice_avg, bl_dice_std = np.mean(baseline_dice, axis=0).squeeze(), np.std(baseline_dice, axis=0).squeeze()
    del baseline_cube

    setattr(main, "weight_exclude", weight_exclude)
    setattr(main, "exclude_target_and_dice_calibration", exclude_target_and_dice_calibration)
    setattr(main, "no_distillation_filter", no_distillation_filter)
    setattr(main, "filter_nc_rr", filter_nc_rr)
    setattr(main, "min_weight_only_for_entropy", min_weight_only_for_entropy)

    hard_exp_name = "monuseg-unet-hard"
    hard_param_set = build_parameter_set(hard_exp_name)
    hard_cube = build_datacube(hard_exp_name)
    reeval_cube = build_datacube("monuseg-reeval-hard")

    index_params, cube_index = create_comp_index(hard_cube, hard_param_set)

    to_plot = [
        "monu_rr", "no_distillation", "weights_mode",
        "weights_consistency_fn", "weights_minimum",
        "weights_neighbourhood", "distil_target_mode"
    ]

    out_params = ["monu_rr", "monu_nc", "sparse_start_after", "n_calibration"]

    param_values = set()

    for _, out_cube in hard_cube.iter_dimensions(*out_params):
        for values, in_cube in out_cube.iter_dimensions(*to_plot):
            param_values.add(values)

    color_map = defaultdict(ColorByCounter(start=1))

    for (monu_rr, monu_nc, ssa, n_calib), out_cube in hard_cube.iter_dimensions(*out_params):
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
            val_dice = get_metric_by_comp_index(in_cube, "val_hard_dice", reeval_cube, index_params, cube_index)
            dice_mean = np.mean(val_dice, axis=0)
            dice_std = np.std(val_dice, axis=0)
            x = np.arange(dice_mean.shape[0])

            plt_with_std(plt.gca(), x, dice_mean, dice_std, label, **color_map[values],
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


if __name__ == "__main__":
    main()