from pprint import pprint

import numpy as np
from itertools import chain
from clustertools import build_datacube
from monuseg_params_and_scores_compare import make_label, plot_current_setup, get_metric, plt_with_std, COLORS
from matplotlib import pyplot as plt


def main(argv):
    baseline_cube = build_datacube("monuseg-unet-baseline")
    baseline_dice = baseline_cube("val_dice")
    bl_dice_avg, bl_dice_std = np.mean(baseline_dice, axis=0).squeeze(), np.std(baseline_dice, axis=0).squeeze()
    del baseline_cube
    cube = build_datacube("monuseg-unet-missing")(weights_mode="constant")
    weights_cube = build_datacube("monuseg-unet-weights")(weights_consistency_fn="quadratic")

    to_plot = [
        "monu_rr", "no_distillation", "weights_mode",
        "weights_consistency_fn", "weights_minimum",
        "weights_neighbourhood"
    ]

    cubes = [cube, weights_cube]

    out_params = ["monu_rr", "monu_nc", "sparse_start_after"]

    param_values = set()

    for _, out_cube in chain(*[c.iter_dimensions(*out_params) for c in cubes]):
        for values, in_cube in out_cube.iter_dimensions(*to_plot):
            param_values.add(values)

    param_val_idxs = {v: i for i, v in enumerate(sorted(list(param_values)))}

    for (monu_rr, monu_nc, ssa), out_cube in chain(*[c.iter_dimensions(*out_params) for c in cubes]):
        plt.figure()
        for_params = {"monu_rr": str(monu_rr), "monu_nc": str(monu_nc), "sparse_start_after": str(ssa)}

        print("> -----------------------------------------------------------------------------")
        print("> -----------------------------------------------------------------------------")
        pprint(out_cube.domain)
        pprint(out_cube.metadata)

        plt_with_std(plt.gca(), np.arange(50), bl_dice_avg, bl_dice_std, label="baseline", color=COLORS[0])

        dice_ymin, dice_ymax = np.min(bl_dice_avg), np.max(bl_dice_avg)

        for values, in_cube in out_cube.iter_dimensions(*to_plot):
            rr, nd, wm, wfn, win, wneigh = values
            if in_cube.diagnose()["Missing ratio"] > 0.0:
                print("inner cube with missing data")
                pprint(in_cube.diagnose())
                pprint(in_cube.domain)
                pprint(in_cube.metadata)
                continue

            label = make_label(wm, {
                "monu_rr": rr, "distillation": int(not eval(nd)),
                "weights_consistency_fn": wfn,
                "weights_minimum": win, "weights_neighbourhood": wneigh
            })

            val_dice = np.array(get_metric("val_dice", in_cube))
            dice_mean = np.mean(val_dice, axis=0)
            dice_std = np.std(val_dice, axis=0)
            x = np.arange(dice_mean.shape[0])

            plt_with_std(plt.gca(), x, dice_mean, dice_std, label, COLORS[(param_val_idxs[values] + 1) % len(COLORS)], do_std=False)

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
        filename = "bl_" + title + ".pdf"
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
