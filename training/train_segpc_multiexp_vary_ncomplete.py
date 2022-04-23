import logging
import os

from clustertools import set_stdout_logging, ParameterSet, Experiment, ConstrainedParameterSet, PrioritizedParamSet
from clustertools.storage import PickleStorage

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import weight_exclude, env_parser


def make_experiment(env_params, segpc_nc, exp_type="self-train"):
    if exp_type not in {"self-train", "baseline-noself", "baseline-nosparse"}:
        raise ValueError("incorrect exp type '{}'".format(exp_type))

    exp_name = "segpc-{}-nc-{}".format(exp_type, segpc_nc)

    param_set = ParameterSet()
    seeds = [604080371, 497403283, 955154649, 486139387, 94829376, 74695622, 900102454, 703192111, 899444528, 676837418]
    epochs = 50
    param_set.add_parameters(dataset="segpc")
    param_set.add_parameters(segpc_ms=seeds)
    param_set.add_parameters(segpc_rr=0.9)
    param_set.add_parameters(segpc_nc=segpc_nc)
    param_set.add_parameters(iter_per_epoch=300)
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=epochs)
    param_set.add_parameters(overlap=0)
    param_set.add_parameters(tile_size=512)
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
        param_set.add_parameters(weights_mode="constant")
        param_set.add_parameters(weights_constant=[0.01, 0.5])
        param_set.add_parameters(weights_consistency_fn="quadratic")
        param_set.add_parameters(weights_minimum=0.0)
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
        prioritized.prioritize('segpc_ms', seed)

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
    os.makedirs(namespace.save_path, exist_ok=True)

    for exp_type in "self-train", "baseline-noself", "baseline-nosparse":
        for segpc_nc in [10, 20, 40, 50, 75, 100, 150, 200]:
            print("Exp '{}' with segpc_nc={}".format(exp_type, segpc_nc))
            experiment = make_experiment(env_params, segpc_nc, exp_type=exp_type)
            environment.run(experiment)
            print()

