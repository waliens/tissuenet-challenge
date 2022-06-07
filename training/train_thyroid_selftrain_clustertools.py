import logging
import os

import numpy as np
from clustertools import set_stdout_logging, ParameterSet, Experiment, ConstrainedParameterSet, PrioritizedParamSet
from clustertools.storage import PickleStorage
from numpy.random import SeedSequence

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import env_parser, weight_exclude, computation_changing_parameters

if __name__ == "__main__":
    set_stdout_logging(logging.INFO)
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    seeds = [int(np.random.default_rng(rng).integers(999999999, size=1)) for rng in SeedSequence(42).spawn(10)]
    param_set.add_parameters(dataset="thyroid")
    param_set.add_parameters(iter_per_epoch=300)
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=50)
    param_set.add_parameters(overlap=0)
    param_set.add_parameters(tile_size=512)
    param_set.add_parameters(lr=0.001)
    param_set.add_parameters(init_fmaps=8)
    param_set.add_parameters(zoom_level=0)
    param_set.add_parameters(rseed=seeds)
    param_set.add_parameters(loss="bce")
    param_set.add_parameters(aug_hed_bias_range=0.025)
    param_set.add_parameters(aug_hed_coef_range=0.025)
    param_set.add_parameters(aug_blur_sigma_extent=0.1)
    param_set.add_parameters(aug_noise_var_extent=0.05)
    param_set.add_parameters(sparse_start_after=10)
    param_set.add_parameters(no_distillation=False)
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode=["constant", "balance_gt_overall", "pred_consistency", "pred_entropy", "pred_merged"])
    param_set.add_parameters(weights_constant=[2.0, 1.0, 0.5, 0.1, 0.01])
    param_set.add_parameters(weights_consistency_fn=["quadratic"])
    param_set.add_parameters(weights_minimum=[0.75, 0.5,  0.1, 0.01, 0.0])
    param_set.add_parameters(weights_neighbourhood=[2])
    param_set.add_parameters(distil_target_mode=["soft", "hard_dice"])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)

    prioritized = PrioritizedParamSet(constrained)
    prioritized.prioritize('distil_target_mode', "hard_dice")
    for seed in seeds:
        prioritized.prioritize('rseed', seed)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("thyroid-self-train", prioritized, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment, excluded={"rseed"})

    # Finally run the experiment
    environment.run(experiment)

