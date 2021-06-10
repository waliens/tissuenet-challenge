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
    cube = build_datacube("thyroid-unet-training-gradual-2")
    start_idx = {v: i for i, v in enumerate(cube.domain["sparse_start_after"])}

    scores = list()
    params = list()
    for (tile_size, loss, sdm, sdr), ext_cube in cube.iter_dimensions("tile_size", "loss", "sparse_data_max", "sparse_data_rate"):
        plt.figure()
        markers_x, markers_y = list(), list()
        for (ssa, ), in_cube in ext_cube.iter_dimensions("sparse_start_after"):
            if in_cube("val_losses") is None:
                continue
            dice = in_cube("val_dice")
            #print(tile_size, loss, ssa, np.min(in_cube("val_losses")), in_cube("val_losses")[-1], np.max(dice), np.argmax(dice), dice[-1])
            start_after = int(ssa)
            dice = np.array(dice)
            n_epochs = dice.shape[0]
            plt.plot(np.arange(n_epochs), dice, c=get_color(start_idx[in_cube.metadata["sparse_start_after"]]), linestyle="-", label="ssa={}".format(ssa))
            if int(start_after) < 30:
                markers_x.append(start_after + 1)
                markers_y.append(dice[start_after + 1])

            scores.append(np.max(dice))
            params.append("loss={}, tsize={}, sdm={}, ssa={}, sdr={}".format(loss, tile_size, sdm, ssa, sdr))

        plt.legend()
        plt.title("size:{} loss:{} sdm:{} sdr:{}".format(tile_size, loss, sdm, sdr))
        plt.scatter(markers_x, markers_y, marker="x", c="red")
        plt.savefig("1_graph_{}_{}_{}_{}.png".format(tile_size, loss, sdm, sdr))

    sdata = sorted(enumerate(params), key=lambda v: scores[v[0]])
    for i, data in sdata:
        print(scores[i], data)

if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
