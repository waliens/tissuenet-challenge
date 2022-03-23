import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, ConstrainedParameterSet
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from generic_train import TrainComputation
from train_monuseg_hard_clustertools import computation_changing_parameters, env_parser, weight_exclude, \
    exclude_target_and_dice_calibration, no_distillation_filter


def min_weight_only_for_entropy(**kwargs):
    is_half = lambda v: (0.49 < v < 0.51)
    is_0 = lambda v: v < 0.01
    wmin = kwargs["weights_minimum"]
    return is_0(wmin) or kwargs["weights_mode"] in {"pred_entropy", "pred_merged"}


if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    seeds = [13315092, 21081788, 26735830, 35788921, 56755036, 56882282, 65682867, 91090292, 93410762, 96319575]

    param_set.add_parameters(dataset="monuseg")
    param_set.add_parameters(monu_ms=seeds[:2])
    param_set.add_parameters(monu_rr=[1.0])
    param_set.add_parameters(monu_nc=[1, 2])
    param_set.add_parameters(iter_per_epoch=150)
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
    param_set.add_parameters(lr_sched_factor=0.5)
    param_set.add_parameters(lr_sched_patience=3)
    param_set.add_parameters(lr_sched_cooldown=5)
    param_set.add_parameters(save_cues=False)
    param_set.add_parameters(sparse_data_rate=1.0)
    param_set.add_parameters(sparse_data_max=1.0)
    param_set.add_parameters(sparse_start_after=[15])
    param_set.add_parameters(no_distillation=[False])
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode=["constant", "pred_consistency", "pred_entropy", "pred_merged"])
    param_set.add_parameters(weights_constant=[1.0])
    param_set.add_parameters(weights_consistency_fn=["quadratic"])
    param_set.add_parameters(weights_minimum=[0.0, 0.1, 0.5])
    param_set.add_parameters(weights_neighbourhood=[2])
    param_set.add_parameters(distil_target_mode=["soft", "hard_dice"])
    param_set.add_parameters(n_calibration=[0, 1])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)
    constrained.add_constraints(exclude_target_and_dice_calibration=exclude_target_and_dice_calibration)
    constrained.add_constraints(no_distillation=no_distillation_filter)
    constrained.add_constraints(min_weight_only_for_entropy=min_weight_only_for_entropy)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-unet-nogt", constrained, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment, excluded={"monu_ms"})

    # Finally run the experiment
    environment.run(experiment)
