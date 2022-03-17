import os

import numpy as np
from clustertools import build_datacube
from clustertools.parameterset import build_parameter_set, CartesianParameterSet

from plot_helpers import get_metric_without_none


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
            if all([inner_cube.metadata[param_name] == str(comp_with_index[comp_idx][param_name]) for param_name in
                    varying]):
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
        idx = str(comp_index[key])
        if idx not in reeval_datacube.domain['comp_index']:
            return None
        metrics.append(reeval_datacube(comp_index=idx)(metric))
    return np.array(metrics)


def base_parameter_set(param_set):
    while not isinstance(param_set, CartesianParameterSet):
        param_set = param_set.param_set
    return param_set


class ExperimentReader(object):
    def __init__(self, exp_name, reeval_exp_name, *seed_params):
        self._cube = build_datacube(exp_name)
        self._param_set = build_parameter_set(exp_name)
        self._reeval_cube = build_datacube(reeval_exp_name)
        self._index_params, self._cube_index = create_comp_index(self._cube, base_parameter_set(self._param_set))
        self._seed_params = set(seed_params)

    def _get_metric(self, metric, src="reeval", **params):
        try:
            param_slice = self._cube(**params)
            valid_cubes = list()
            other_params = set(param_slice.parameters).difference(self._seed_params)
            metric_value = None
            for in_values, in_cube in param_slice.iter_dimensions(*other_params):
                if in_cube.diagnose()["Missing ratio"] >= 1.0:
                    continue
                valid_cubes.append({k: v for k, v in zip(other_params, in_values)})
                if src == "reeval":
                    metric_value = get_metric_by_comp_index(in_cube, metric, self._reeval_cube, self._index_params,
                                                            self._cube_index)
                else:
                    metric_value = get_metric_without_none(in_cube, metric)
            if len(valid_cubes) > 1:
                print(other_params, "for", params)
                raise ValueError("more than one param comb found for metric {}".format(metric))
            if metric_value is None:
                return None

            return metric_value
        except IndexError:
            return None
        except KeyError:
            return None

    def get_reeval_metric(self, metric, **params):
        return self._get_metric(metric, **params)

    def get_metric(self, metric, **params):
        return self._get_metric(metric, **params, src="original")


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
        return "$" + "{}".format(float(params["weights_minimum"])) + \
            "$ & $" + "{}".format(int(params["weights_neighbourhood"])) + \
            "$ & $" + "{}".format("|\\cdot|" if params["weights_consistency_fn"] == "absolute" else "\\cdot^2") + "$"
    return "& &"


def get_super_row(current_mode, n_columns):
    name = {
        "constant": "Constant ($C$)",
        "balance_gt": "Balance",
        "pred_entropy": "Entropy - $w_{\\text{min}}$",
        "pred_consistency": "Consistency - $\\eta, c(y_1, y_2)$",
        "pred_merged": "Merged - $w_{\\text{min}}, \\eta, c(y_1, y_2)$",
        "none": "No self-training"
    }[current_mode]
    return os.linesep.join(["\\hline", "\\multicolumn{" + str(n_columns) + "}{|l|}{" + name + "} \\\\", "\\hline"])


def get_column_headers(columns, total_train_img):
    rows_content = [
        ["\\multicolumn{3}{|c|}{$|\\mathcal{D}_l|/|\\mathcal{D}_s|$}"],
        ["\\multicolumn{3}{|c|}{$\\rho$}"],
        ["\\multicolumn{3}{|c|}{$|\\mathcal{D}_{cal}|$}"],
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
        n_cal = int(column["n_calibration"])
        rows_content[0].append("{}/{}".format(nc - n_cal, total_train_img - nc))
        rows_content[1].append("{:3d}\\%".format(int(rr * 100)))
        rows_content[2].append(str(n_cal))

    return os.linesep.join([(" & ".join(row) + "\\\\") for row in rows_content])


def plot_table(rows, columns, total_train_imgs):
    print("\\begin{table*}")
    print("\\begin{tabular}{|ccc|" + "c" * len(columns) + "|}")
    print(get_column_headers(columns, total_train_imgs))

    current_mode = None
    for row in rows:
        if 'weights_mode' not in row or current_mode != row["weights_mode"]:
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
