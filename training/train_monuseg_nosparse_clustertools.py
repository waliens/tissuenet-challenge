import itertools
import os
from collections import defaultdict

from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, ConstrainedParameterSet, \
    PrioritizedParamSet
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import env_parser, computation_changing_parameters

if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    seeds = [13315092, 21081788, 26735830, 35788921, 56755036, 56882282, 65682867, 91090292, 93410762, 96319575]
    epochs = 20
    param_set.add_parameters(dataset="monuseg")
    param_set.add_parameters(monu_ms=seeds)
    param_set.add_parameters(monu_rr=0.9)
    param_set.add_parameters(monu_nc=2)
    param_set.add_parameters(iter_per_epoch=100)
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
    param_set.add_parameters(sparse_start_after=epochs)
    param_set.add_parameters(no_distillation=True)
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode="constant")
    param_set.add_parameters(weights_constant=1.0)
    param_set.add_parameters(weights_consistency_fn="quadratic")
    param_set.add_parameters(weights_minimum=0.0)
    param_set.add_parameters(weights_neighbourhood=2)
    param_set.add_parameters(distil_target_mode="soft")


    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-baseline-nosparse", param_set, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment, excluded={"monu_ms"})

    # Finally run the experiment
    environment.run(experiment)

