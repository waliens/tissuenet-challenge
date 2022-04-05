import os

from clustertools import set_stdout_logging, ParameterSet, Experiment, ConstrainedParameterSet
from clustertools.storage import PickleStorage

from generic_train import TrainComputation
from train_monuseg_selftrain_clustertools import env_parser, computation_changing_parameters
from train_segpc_selftrain_clustertools import read_segpc_datasets

if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    segpc_ms, segpc_rr, segpc_nc, segpc_constraints = read_segpc_datasets(namespace.data_path)

    param_set = ParameterSet()
    param_set.add_parameters(dataset="segpc")
    param_set.add_parameters(segpc_ms=segpc_ms)
    param_set.add_parameters(segpc_rr=0.9)
    param_set.add_parameters(segpc_nc=30)
    param_set.add_parameters(iter_per_epoch=300)
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=50)
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
    param_set.add_parameters(sparse_start_after=10)
    param_set.add_parameters(no_distillation=True)
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode="constant")
    param_set.add_parameters(weights_constant=1.0)
    param_set.add_parameters(weights_consistency_fn="quadratic")
    param_set.add_parameters(weights_minimum=0.0)
    param_set.add_parameters(weights_neighbourhood=2)
    param_set.add_parameters(distil_target_mode="soft")

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(**segpc_constraints)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("segpc-baseline-noself", constrained, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment)

    # Finally run the experiment
    environment.run(experiment)

