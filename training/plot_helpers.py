import numpy as np


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

LINESTYLES = ["-", ":", "--", "-."]


def get_color(i):
    return COLORS[i % len(COLORS)]


def get_metric(metric_name, cube):
    return [p_cube(metric_name) for p_values, p_cube in cube.iter_dimensions(*cube.parameters)]


def plt_with_std(ax, x, mean, std, label, color, do_std=True, alpha=0.6, linestyle="-"):
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle)
    if do_std:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def get_metric_without_none(cube, metric):
    data = []
    for _, in_cube in cube.iter_dimensions(*cube.parameters):
        if in_cube.diagnose()["Missing ratio"] <= 0.0:
            data.append(in_cube(metric))
    return np.array(data)


def cube_key(cube, *params):
    if len(params) == 0:
        params = list(cube.metadata.keys()) + list(cube.domain.keys())
    return tuple(map(lambda p: str(cube.metadata[p]), params))


def create_comp_index(cube, param_set):
    varying = sorted(cube.domain.keys())
    comp_with_index = {k: v for k, v in param_set}
    remaining_comp = set(comp_with_index.keys())
    cube_index = dict()

    for v, inner_cube in cube.iter_dimensions(*varying):
        if inner_cube.diagnose()["Missing ratio"] > 0:
            continue

        selected_idx = None
        for comp_idx in remaining_comp:
            if all([inner_cube.metadata[param_name] == str(comp_with_index[comp_idx][param_name]) for param_name in varying]):
                selected_idx = comp_idx
                break
        if selected_idx is None:
            raise ValueError("'{}' not found".format(v))
        remaining_comp.remove(selected_idx)
        cube_index[cube_key(inner_cube, *varying)] = selected_idx

    return varying, cube_index


def get_metric_by_comp_index(cube, metric, reeval_datacube, index_params, comp_index):
    metrics = list()
    for _, in_cube in cube.iter_dimensions(*cube.domain.keys()):
        key = cube_key(in_cube, *index_params)
        if key not in comp_index:
            continue
        idx = comp_index[key]
        metrics.append(reeval_datacube(comp_index=str(idx))(metric))
    return np.array(metrics)


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


class ColorByCounter(object):
    def __init__(self, start=0):
        self._COLORS = COLORS
        self._LINESTYLES = LINESTYLES
        self._start = start
        self._counter = 0

    def __call__(self, *args, **kwargs):
        curr_counter = self._counter
        color_idx = (self._start + curr_counter) % len(self._COLORS)
        linestyle_idx = ((self._start + curr_counter) // len(self._COLORS)) % len(self._LINESTYLES)
        self._counter += 1
        return {"color": self._COLORS[color_idx], "linestyle": self._LINESTYLES[linestyle_idx]}
