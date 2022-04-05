import os
from collections import defaultdict
from itertools import product

import numpy as np
from clustertools import build_datacube
from clustertools.experiment import load_computation
from clustertools.storage import PickleStorage
from clustertools.parameterset import build_parameter_set, CartesianParameterSet

from plot_helpers import get_metric_without_none


def cube_key(comp_params, *params):
    return tuple(map(lambda p: str(comp_params[p]), params))


def build_domain_and_metadata(comp_params):
    dd = defaultdict(set)
    for curr_params in comp_params:
        for p, v in curr_params.items():
            dd[p].add(v)
    return {k: list(v) for k, v in dd.items() if len(v) > 1}, {k: list(v)[0] for k, v in dd.items() if len(v) == 1}


def create_comp_index(exp_map):
    exp_domain, exp_metadata = build_domain_and_metadata([v[0] for v in exp_map.values()])
    varying = sorted(exp_domain.keys())
    cube_index = dict()

    for comp_idx, (comp_params, comp_results) in exp_map.items():
        cube_index[cube_key(comp_params, *varying)] = comp_idx

    return varying, (exp_domain, exp_metadata), cube_index


# def get_metric_by_comp_index(cube, metric, reeval_datacube, index_params, comp_index):
#     metrics = list()
#     for _, in_cube in cube.iter_dimensions(*cube.domain.keys()):
#         key = cube_key(in_cube, *index_params)
#         if key not in comp_index:
#             continue
#         idx = str(comp_index[key])
#         if idx not in reeval_datacube.domain['comp_index']:
#             return None
#         metrics.append(reeval_datacube(comp_index=idx)(metric))
#     return np.array(metrics)


def base_parameter_set(param_set):
    while not isinstance(param_set, CartesianParameterSet):
        param_set = param_set.param_set
    return param_set


def load_indexes(exp_name):
    storage = PickleStorage(exp_name)
    result_folder = os.path.join(storage.folder, "results")
    return [fname.split(".")[0].rsplit("-")[-1] for fname in os.listdir(result_folder)]


class ExperimentReader(object):
    def __init__(self, exp_name):
        exp_indexes = load_indexes(exp_name)
        self._exp_map = {idx: load_computation(exp_name, int(idx)) for idx in exp_indexes}
        self._index_params, (self._exp_domain, self._exp_metadata), self._params_to_exp_index = create_comp_index(self._exp_map)

    def _get_metric(self, metric, **params):
        domain_keys = set(self._exp_domain.keys())
        param_keys = set(params.keys())
        missing_params = domain_keys.difference(param_keys)
        non_varying = param_keys.difference(domain_keys)
        if any([str(self._exp_metadata[p]) != params[p] for p in non_varying]):
            return None
        results = list()
        for m_values in product(*[self._exp_domain[p] for p in missing_params]):
            all_params = {**params, **{mp: mv for mp, mv in zip(missing_params, m_values)}}
            key = cube_key(all_params, *self._index_params)
            if key not in self._params_to_exp_index:
                continue
            exp_index = self._params_to_exp_index[key]
            results.append(self._exp_map[exp_index][1][metric])
        if len(results) == 0:
            return None
        return np.array(results)

    def get_metric(self, metric, **params):
        return self._get_metric(metric, **params)


def get_row_header(mode, **params):
    if mode == "constant":
        return "\\multicolumn{3}{|c|}{$"+"{}".format(float(params["weights_constant"]))+"$}"
    elif mode == "balance_gt":
        return "& &"
    elif mode == "pred_entropy":
        return "\\multicolumn{3}{|c|}{$" + "{}".format(float(params["weights_minimum"])) + "$}"
    elif mode == "pred_consistency":
        return "$"+"{}".format(int(params["weights_neighbourhood"])) + \
            "$ & \\multicolumn{2}{c|}{$" + \
            "{}".format("|\\cdot|" if params["weights_consistency_fn"] == "absolute" else "\\cdot^2")+"$}"
    elif mode == "pred_merged":
        return "$" + "{}".format(float(params.get("weights_minimum", "0"))) + \
            "$ & $" + "{}".format(int(params["weights_neighbourhood"])) + \
            "$ & $" + "{}".format("|\\cdot|" if params["weights_consistency_fn"] == "absolute" else "\\cdot^2") + "$"
    elif "type" in params and params["type"] == "bl-upper":
        return "\\multicolumn{3}{|c|}{$|\\mathcal{D}_s| = 0$}"
    elif "type" in params and params["type"] == "bl-noself":
        return "\\multicolumn{3}{|c|}{$\\mathcal{D}_l \cup \\mathcal{D}_s$}"
    elif "type" in params and params["type"] == "bl-nosparse":
        return "\\multicolumn{3}{|c|}{$\\mathcal{D}_l$ only}"
    else:
        return params["type"] + "& &"


def get_super_row(current_mode, n_columns):
    name = {
        "constant": "Constant ($C$)",
        "balance_gt": "Balance",
        "pred_entropy": "Entropy - $w_{\\text{min}}$",
        "pred_consistency": "Consistency - $\\eta, c(y_1, y_2)$",
        "pred_merged": "Merged - $w_{\\text{min}}, \\eta, c(y_1, y_2)$",
        "none": "Baselines"
    }[current_mode]
    return os.linesep.join(["\\hline", "\\multicolumn{" + str(n_columns) + "}{|l|}{" + name + "} \\\\", "\\hline"])


def get_column_headers(columns, total_train_img):
    rows_content = [
        ["\\multicolumn{3}{|c|}{$|\\mathcal{D}_l|/|\\mathcal{D}_s|$}"],
        ["\\multicolumn{3}{|c|}{$\\rho$}"]
    ]

    for _, column in columns:
        ncs = [k for k in column.keys() if k.endswith("_nc")]
        nc = 0
        if len(ncs) > 0:
            nc_key = ncs[0]
            nc = int(column[nc_key])
        rrs = [k for k in column.keys() if k.endswith("_rr")]
        rr = 1.0
        if len(rrs) > 0:
            rr_key = rrs[0]
            rr = float(column[rr_key])
        rows_content[0].append("{}/{}".format(nc, total_train_img - nc))
        rows_content[1].append("{:3d}\\%".format(int(rr * 100)))

    return os.linesep.join([(" & ".join(row) + "\\\\") for row in rows_content])


def plot_table(rows, columns, total_train_imgs):
    print("\\begin{table*}")
    print("\\begin{tabular}{|ccc|" + "c" * len(columns) + "|}")
    print(get_column_headers(columns, total_train_imgs))

    current_mode = None
    for row in rows:
        if row.get("weights_mode", "none") != current_mode:
            current_mode = row.get('weights_mode', 'none')
            print(get_super_row(current_mode, len(columns) + 3))

        row_items = list()
        for metric_fn, column in columns:
            values = metric_fn(**row, **column)
            if values is None:
                row_items.append(" " * 18)
            else:
                values = np.array(values)
                if values.ndim == 3:
                    values = values.squeeze()
                avg, std = np.mean(values, axis=0), np.std(values, axis=0)
                idx = np.argmax(avg)
                row_items.append("{:2.2f} ± {:2.2f}".format(avg[idx] * 100, std[idx] * 100))

        if len("".join(row_items).strip()) > 0:
            print(get_row_header(current_mode, **row).rjust(40), " & ", " & ".join(row_items), "\\\\")

    print("\\end{tabular}")
    print("\\end{table*}")
