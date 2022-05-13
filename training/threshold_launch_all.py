import logging
import os

from clustertools import set_stdout_logging, ParameterSet, Experiment
from clustertools.storage import PickleStorage

from generic_threshold import TuneThresholdComputation
from train_monuseg_selftrain_clustertools import env_parser


def launch(exp_name, comp_indexes, **env_params):
    thresh_exp_name = exp_name + "-thresh"
    comp_indexes = sorted(comp_indexes)

    param_set = ParameterSet()
    param_set.add_parameters(nbs="1,2,3,4")
    param_set.add_parameters(train_exp=exp_name)
    for idx in comp_indexes:
        param_set.add_parameters(comp_index=idx)
        param_set.add_separator()

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TuneThresholdComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    return Experiment(thresh_exp_name, param_set, make_build_fn(**env_params))


def get_indexes(exp_name):
    storage = PickleStorage(exp_name)
    result_folder = os.path.join(storage.folder, "results")
    if not os.path.exists(result_folder):
        print(result_folder, "does not exist")
        return []
    indexes = sorted([int(fname.rsplit(".")[-2].rsplit("-")[-1]) for fname in os.listdir(result_folder)])
    return indexes


def main(argv):
    set_stdout_logging(logging.INFO)

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    datasets = {
        "monuseg": {
            "nc": [1, 3, 4, 5, 10, 15],
            "rr": [1.0, 0.95, 0.85, 0.8, 0.75, 0.60, 0.5, 0.975, 0.99, 0.25]
        },
        "glas": {
            "nc": [2, 4, 16, 24, 32, 40, 60],
            "rr": [1.0, 0.99, 0.975, 0.95, 0.85, 0.8, 0.75, 0.6, 0.5, 0.25]
        },
        "segpc": {
            "nc": [10, 20, 40, 50, 75, 100, 150, 200],
            "rr": [1.0, 0.95, 0.85, 0.8, 0.75, 0.60, 0.5, 0.25]
        }
    }

    exp_types = ["self-train", "baseline-noself", "baseline-nosparse"]

    total = 0
    for dataset, params in datasets.items():
        env_params["save_path"] = "/home/rmormont/models/{}-unet/varying".format(dataset)
        env_params["data_path"] = "/scratch/users/rmormont/{}".format(dataset)
        for exp_type in exp_types:
            for exp_study in ["rr", "nc"]:
                for pval in params[exp_study]:
                    exp_name = "{}-{}-{}-{}".format(dataset, exp_type, exp_study, ("{:1.4f}".format(pval) if exp_study == "rr" else pval))
                    idxs = get_indexes(exp_name)
                    if len(idxs) == 0:
                        continue
                    total += len(idxs)
                    print(exp_name, len(idxs))
                    experiment = launch(exp_name, idxs, **env_params)
                    environment.run(experiment)

        for exp, folder in [
            ("baseline-upper-noval", "upper_noval"),
            ("self-train", "selftrain"),
            ("baseline-noself", "noself"),
            ("baseline-nosparse", "nosparse")
        ]:
            env_params["save_path"] = "/home/rmormont/models/{}-unet/{}".format(dataset, folder)
            exp_name = "{}-{}".format(dataset, exp)
            idxs = get_indexes(exp_name)
            if len(idxs) == 0:
                continue
            total += len(idxs)
            experiment = launch(exp_name, idxs, **env_params)
            environment.run(experiment)

    print("total:", total)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
