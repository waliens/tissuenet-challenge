import json
import os
from abc import abstractmethod, abstractproperty
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
    return [fname.rsplit(".")[-2].rsplit("-")[-1] for fname in os.listdir(result_folder)]


class BaseExperimentReader(object):

    @abstractmethod
    def get_metric(self, metric, **params):
        pass

    @abstractmethod
    def get_computations(self, **params):
        pass

    @property
    @abstractmethod
    def exp_name(self):
        pass


class ExperimentReader(BaseExperimentReader):
    def __init__(self, exp_name):
        exp_indexes = load_indexes(exp_name)
        self._exp_name = exp_name
        self._exp_map = {idx: load_computation(exp_name, int(idx)) for idx in exp_indexes}
        self._index_params, (self._exp_domain, self._exp_metadata), self._params_to_exp_index = create_comp_index(self._exp_map)

    @property
    def exp_name(self):
        return self._exp_name

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
            if metric not in self._exp_map[exp_index][1]:
                continue
            results.append(self._exp_map[exp_index][1][metric])
        if len(results) == 0:
            return None
        return np.array(results)

    def get_computations(self, **params):
        domain_keys = set(self._exp_domain.keys())
        param_keys = set(params.keys())
        missing_params = domain_keys.difference(param_keys)
        non_varying = param_keys.difference(domain_keys)
        if any([str(self._exp_metadata[p]) != params[p] for p in non_varying]):
            return []
        for m_values in product(*[self._exp_domain[p] for p in missing_params]):
            all_params = {**params, **{mp: mv for mp, mv in zip(missing_params, m_values)}}
            key = cube_key(all_params, *self._index_params)
            if key not in self._params_to_exp_index:
                continue
            exp_index = self._params_to_exp_index[key]
            yield exp_index, self._exp_map[exp_index]

    def get_metric(self, metric, **params):
        return self._get_metric(metric, **params)


class FollowUpExperimentReader(BaseExperimentReader):
    def __init__(self, base_experiment, suffix):
        self._base_experiment = base_experiment
        self._suffix = suffix
        self._follow_up = ExperimentReader(self.exp_name)

    def get_metric(self, metric, **params):
        parent_metric = self._base_experiment.get_metric(metric, **params)
        if parent_metric is not None:
            return parent_metric
        values = list()
        metric_key = self._suffix + metric
        for idx, (params, results) in self.get_computations(**params):
            if metric_key not in results:
                continue
            values.append(results[metric_key])
        if len(values) == 0:
            return None
        return np.array(values)

    def get_computations(self, **params):
        for idx, (params, metrics) in self._base_experiment.get_computations(**params):
            fu_comps = list(self._follow_up.get_computations(
                comp_index=idx,
                train_exp=self._base_experiment.exp_name))
            if len(fu_comps) != 1:
                raise ValueError("unexpected number of follow up computations: {}".format(len(fu_comps)))
            _, (_, fu_metrics) = fu_comps[0]
            yield idx, (params, {**metrics, **{(self._suffix + k): v for k, v in fu_metrics.items()}})

    @property
    def exp_name(self):
        return self._base_experiment.exp_name + self._suffix


def get_row_header(mode, **params):
    if mode == "constant":
        return "\\multicolumn{3}{|c|}{$"+"{}".format(float(params["weights_constant"]))+"$}"
    elif mode.startswith("balance_gt"):
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
        "balance_gt_overall": "Balance (overall)",
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


class ScoreReader(object):
    def __init__(self, exp_name, metric, stats, param="rr", **params):
        self._exp_name = exp_name
        self._base_reader = ExperimentReader(self._exp_name)
        self._reader = FollowUpExperimentReader(self._base_reader, "-thresh")
        self._params = params
        self._metric = metric
        self._param = param
        self._stats = stats

    def get_xy(self):
        all_results = list(self._reader.get_computations(**self._params))
        if len(all_results) == 0:
            return -1, None, None
            # raise ValueError("no comb {}: {}".format(self._exp_name, self._params))
        indexes, computations = zip(*all_results)
        params, results = computations[0]
        values = self._reader.get_metric(self._metric, **self._params)
        if np.array(values).ndim > 1:
            values = np.array(values)[:, -1]
        return params[self._param_name], np.mean(values), np.std(values)

    def get_scatter(self, x_stat):
        all_results = list(self._reader.get_computations(**self._params))
        if len(all_results) == 0:
            return np.array([]), np.array([])
            # raise ValueError("no comb {}: {}".format(self._exp_name, self._params))
        _, computations = zip(*all_results)

        x, y = list(), list()
        for params, results in computations:
            nc = params[self._param_prefix + "nc"]
            rr = params[self._param_prefix + "rr"]
            ms = params[self._param_prefix + "ms"]
            key = "_".join(map(str, [self._dataset, ms, "{:0.4f}".format(float(rr)), nc]))
            x.append(self._stats[key]["stats"][x_stat])
            if self._metric in results:
                values = results[self._metric]
            else:
                values = results[self._reader._suffix + self._metric]
            if hasattr(values, "__len__"):
                value = values[-1]
            else:
                value = values
            y.append(value)

        return np.array(x), np.array(y)

    @property
    def _param_prefix(self):
        if "monuseg" in self._exp_name:
            return "monu_"
        elif "segpc" in self._exp_name:
            return "segpc_"
        elif "glas" in self._exp_name:
            return "glas_"
        else:
            raise ValueError("invalid experiment")

    @property
    def _dataset(self):
        if "monuseg" in self._exp_name:
            return "monuseg"
        elif "segpc" in self._exp_name:
            return "segpc"
        elif "glas" in self._exp_name:
            return "glas"
        else:
            raise ValueError("invalid experiment")

    @property
    def _param_name(self):
        return self._param_prefix + self._param
