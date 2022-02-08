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


def main(argv):
    cube = build_datacube("thyroid-unet-training-weights")
    start_idx = {v: i for i, v in enumerate(cube.domain["sparse_start_after"])}

    "sparse_start_after"


    "rseed"

    scores = list()
    stds = list()
    final_scores = list()
    final_stds = list()
    params = list()
    for (no_distill, weights, w_cst, w_cst_fn, w_min, w_neigh), ext_cube in cube.iter_dimensions("no_distillation", "weights_mode", "weights_constant", "weights_consistency_fn", "weights_minimum", "weights_neighbourhood"):
        plt.figure()
        markers_x, markers_y = list(), list()

        title = "no_distill:{} weights:{} w_cst:{} w_cst_fn:{} w_min:{} w_neigh:{}".format(no_distill, weights, w_cst,
                                                                                           w_cst_fn, w_min, w_neigh)

        for (ssa, ), in_cube in ext_cube.iter_dimensions("sparse_start_after"):
            dices = list()
            for (seed, ), seed_cube in in_cube.iter_dimensions("rseed"):
                if seed_cube("val_losses") is None:
                    continue
                dice = seed_cube("val_dice")
                dice = np.array(dice)
                dices.append(dice)
                # print(tile_size, loss, ssa, nogt, nodst, np.min(in_cube("val_losses")), in_cube("val_losses")[-1],
                #           np.max(dice), np.argmax(dice), dice[-1])
                # start_after = int(ssa)
                # n_epochs = dice.shape[0]
                # plt.plot(np.arange(n_epochs), dice, c=get_color(start_idx[in_cube.metadata["sparse_start_after"]]),
                #           linestyle="-", label="ssa={}".format(ssa))

            if len(dices) == 0:
                continue

            dices = np.array(dices)
            avg_dices = np.mean(dices, axis=0)
            start_after = int(ssa)
            if start_after < 49:
                markers_x.append(start_after + 1)
                markers_y.append(avg_dices[start_after + 1])

            params.append(title + " ssa:{}".format(ssa))
            scores.append(np.mean(np.max(dices, axis=1)))
            stds.append(np.std(np.max(dices, axis=1)))
            final_scores.append(np.mean(dices[:, -1]))
            final_stds.append(np.std(dices[:, -1]))
            color = get_color(start_idx[in_cube.metadata["sparse_start_after"]])
            plt.plot(np.arange(avg_dices.shape[0]), avg_dices, c=color, linestyle="-", label="ssa={}".format(ssa))

        if len(markers_x) == 0:
            continue

        plt.legend()
        plt.title(title)
        plt.scatter(markers_x, markers_y, marker="x", c="red")
        plt.savefig("graphs/1_graph_{}_{}_{}_{}_{}_{}.png".format(no_distill, weights, w_cst, w_cst_fn, w_min, w_neigh))
        plt.close()

    sdata = sorted(enumerate(params), key=lambda v: scores[v[0]])
    for i, data in sdata:
        print("{:0.4f}".format(scores[i]), "+-", "{:1.3e}".format(stds[i]), "|", "{:0.4f}".format(final_scores[i]), "+-", "{:1.3e}".format(final_stds[i]), data)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
