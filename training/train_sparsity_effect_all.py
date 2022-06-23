import itertools
import logging
import os
import re

from clustertools import ParameterSet, Experiment, ConstrainedParameterSet, PrioritizedParamSet, set_stdout_logging
from clustertools.storage import PickleStorage

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import weight_exclude, env_parser


def make_experiment(env_params, dataset, dataset_key, seeds, nc, rr, exp_type="self-train", **dataset_params):
    if exp_type not in {"self-train", "baseline-noself", "baseline-nosparse"}:
        raise ValueError("incorrect exp type '{}'".format(exp_type))

    exp_name = "{}-{}-nc-{}-rr-{}".format(dataset, exp_type, nc, rr)

    param_set = ParameterSet()
    epochs = dataset_params["epochs"]
    param_set.add_parameters(dataset=dataset)
    param_set.add_parameters(**{"{}_ms".format(dataset_key): seeds})
    param_set.add_parameters(**{"{}_rr".format(dataset_key): rr})
    param_set.add_parameters(**{"{}_nc".format(dataset_key): nc})
    param_set.add_parameters(iter_per_epoch=dataset_params["iter_per_epoch"])
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=epochs)
    param_set.add_parameters(overlap=0)
    param_set.add_parameters(tile_size=dataset_params["tile_size"])
    param_set.add_parameters(lr=0.001)
    param_set.add_parameters(init_fmaps=8)
    param_set.add_parameters(zoom_level=0)
    param_set.add_parameters(rseed=42)
    param_set.add_parameters(loss="bce")
    param_set.add_parameters(aug_hed_bias_range=0.025)
    param_set.add_parameters(aug_hed_coef_range=0.025)
    param_set.add_parameters(aug_blur_sigma_extent=0.1)
    param_set.add_parameters(aug_noise_var_extent=0.05)
    param_set.add_parameters(sparse_start_after=10 if exp_type != "baseline-nosparse" else epochs)
    param_set.add_parameters(no_distillation=False if exp_type == "self-train" else True)
    param_set.add_parameters(no_groundtruth=False)

    if exp_type == "self-train":
        param_set.add_parameters(weights_mode="pred_entropy")
        param_set.add_parameters(weights_constant=1.0)
        param_set.add_parameters(weights_minimum=0.1)
        param_set.add_parameters(weights_consistency_fn="quadratic")
        param_set.add_parameters(weights_neighbourhood=2)
        param_set.add_parameters(distil_target_mode="hard_dice")
    else:
        param_set.add_parameters(weights_mode="constant")
        param_set.add_parameters(weights_constant=1.0)
        param_set.add_parameters(weights_consistency_fn="quadratic")
        param_set.add_parameters(weights_minimum=0.0)
        param_set.add_parameters(weights_neighbourhood=2)
        param_set.add_parameters(distil_target_mode="soft")

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)

    prioritized = PrioritizedParamSet(constrained)
    for seed in seeds:
        prioritized.prioritize('{}_ms'.format(dataset_key), seed)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    return Experiment(exp_name, prioritized, make_build_fn(**env_params))


if __name__ == "__main__":
    set_stdout_logging(logging.INFO)

    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    params = {
        "monuseg": {
            "iter_per_epoch": 100,
            "epochs": 20,
            "tile_size": 512
        },
        "segpc": {
            "iter_per_epoch": 300,
            "epochs": 30,
            "tile_size": 512
        },
        "glas": {
            "iter_per_epoch": 225,
            "epochs": 20,
            "tile_size": 384
        }
    }

    sparsity_params = {
        "monuseg": {
            "rr": [0.5, 0.25],
            "nc": [5, 10, 15]
        },
        "glas": {
            "rr": [0.5, 0.25],
            "nc": [16, 32, 40]
        },
        "segpc": {
            "rr": [0.5, 0.25],
            "nc": [50, 100, 150]
        }
    }

    SEEDS = {
        "monuseg": [13315092, 21081788, 26735830, 35788921, 56755036, 56882282, 65682867, 91090292, 93410762, 96319575],
        "segpc": [486139387, 497403283, 604080371, 676837418, 703192111, 74695622, 899444528, 900102454, 94829376, 955154649],
        "glas": [486139387, 497403283, 604080371, 676837418, 703192111, 74695622, 899444528, 900102454, 94829376, 955154649]
    }

    all_experiments = list()
    for dataset, d_key in [("monuseg", "monu"), ("segpc", "segpc"), ("glas", "glas")]:
        env_params["save_path"] = re.sub(r"(monuseg|segpc|glas|DATASET)", dataset, env_params["save_path"])
        env_params["data_path"] = re.sub(r"(monuseg|segpc|glas|DATASET)", dataset, env_params["data_path"])
        os.makedirs(namespace.save_path, exist_ok=True)
        sparsity_dict = sparsity_params[dataset]
        for exp_type in ["self-train", "baseline-noself", "baseline-nosparse"]:
            for rr, nc in itertools.product(sparsity_dict["rr"], sparsity_dict["nc"]):
                if "nosparse" in exp_type and not 0.49 < rr < 0.51:
                    continue

                print(".", end="", flush=True)
                actual_params = {**params[dataset]}
                if "nosparse" not in exp_type:
                    actual_params["epochs"] = 50
                experiment = make_experiment(
                    env_params, dataset, d_key, SEEDS[dataset], nc, rr, exp_type=exp_type, **actual_params)
                all_experiments.append(experiment)
    print()
    total = sum([len(exp.parameter_set) for exp in all_experiments])
    print("Total: {} computations".format(total))

    for exp in all_experiments:
        exp.monitor.aborted_to_launchable()

    for exp in all_experiments:
        print(">", exp.exp_name)
        environment.run(exp)

