import logging
import os

from clustertools import set_stdout_logging, ParameterSet, Experiment, ConstrainedParameterSet, PrioritizedParamSet
from clustertools.storage import PickleStorage

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import env_parser, weight_exclude, computation_changing_parameters

if __name__ == "__main__":
    set_stdout_logging(logging.INFO)
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    seeds = [486139387, 497403283, 604080371, 676837418, 703192111, 74695622, 899444528, 900102454, 94829376, 955154649]
    param_set.add_parameters(dataset="glas")
    param_set.add_parameters(glas_ms=seeds)
    param_set.add_parameters(glas_rr=0.9)
    param_set.add_parameters(glas_nc=8)
    param_set.add_parameters(iter_per_epoch=225)
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=50)
    param_set.add_parameters(overlap=0)
    param_set.add_parameters(tile_size=384)
    param_set.add_parameters(lr=0.001)
    param_set.add_parameters(init_fmaps=8)
    param_set.add_parameters(zoom_level=0)
    param_set.add_parameters(rseed=42)
    param_set.add_parameters(loss="bce")
    param_set.add_parameters(aug_hed_bias_range=0.025)
    param_set.add_parameters(aug_hed_coef_range=0.025)
    param_set.add_parameters(aug_blur_sigma_extent=0.1)
    param_set.add_parameters(aug_noise_var_extent=0.05)
    param_set.add_parameters(sparse_start_after=10)
    param_set.add_parameters(no_distillation=False)
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode=["constant", "balance_gt_overall", "balance_gt", "pred_consistency", "pred_entropy", "pred_merged"])
    param_set.add_parameters(weights_constant=[3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01])
    param_set.add_parameters(weights_consistency_fn=["quadratic"])
    param_set.add_parameters(weights_minimum=[0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0])
    param_set.add_parameters(weights_neighbourhood=[2])
    param_set.add_parameters(distil_target_mode=["soft", "hard_dice"])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)

    prioritized = PrioritizedParamSet(constrained)
    for seed in seeds:
        prioritized.prioritize('glas_ms', seed)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("glas-self-train", prioritized, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment, excluded={"glas_ms"})

    # Finally run the experiment
    environment.run(experiment)

