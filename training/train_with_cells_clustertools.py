import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, Computation, ConstrainedParameterSet
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from train_with_cells import TrainComputation


def env_parser():
    parser = CTParser()
    parser.add_argument("--save_path", dest="save_path")
    parser.add_argument("--data_path", "--data_path", dest="data_path")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    _ = Cytomine._add_cytomine_cli_args(parser.parser)
    return parser


def exclude_no_new_data(**kwargs):
    return kwargs["sparse_start_after"] < 50 or (kwargs["sparse_start_after"] == 50 and kwargs["sparse_data_max"] > .99 and .99 < kwargs["sparse_data_rate"] < 1.01)


def at_least_one_source_of_annot(**kwargs):
    return not (kwargs.get("no_distillation") and kwargs.get("no_groundtruth"))


def exclude_no_groundtruth_no_pretraining_data(**kwargs):
    return not (kwargs.get("no_groundtruth") and kwargs.get("sparse_start_after") == -1)


def weight_exclude(**kwargs):
    wconstant_is_one = 0.99 < kwargs["weights_constant"] < 1.01
    wmode_is_not_constant = kwargs["weights_mode"] != "constant"
    wmode_is_constant = not wmode_is_not_constant
    min_weight_is_zero = kwargs.get("weights_minimum") < 0.01
    constant_and_is_one = (wmode_is_constant and wconstant_is_one)
    return (not kwargs.get("no_distillation") or (constant_and_is_one and min_weight_is_zero)) \
        and (not kwargs.get("no_groundtruth") or (constant_and_is_one and min_weight_is_zero)) \
        and (wmode_is_constant or wconstant_is_one) \
        and (kwargs.get("weights_mode") in {"pred_consistency", "pred_merged"} or
                (kwargs.get("weights_consistency_fn") == "absolute" and kwargs.get("weights_neighbourhood") == 1)) \
        and (kwargs.get("loss") == "bce" or (constant_and_is_one and min_weight_is_zero)) \
        and (kwargs.get("sparse_start_after") < 50 or (constant_and_is_one and min_weight_is_zero)) \
        and (kwargs.get("weights_mode") not in {"constant", "gt_balance"} or min_weight_is_zero)


if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    #os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    param_set.add_parameters(batch_size=[8])
    param_set.add_parameters(epochs=[50])
    param_set.add_parameters(overlap=[0])
    param_set.add_parameters(tile_size=[512])
    param_set.add_parameters(lr=[0.001])
    param_set.add_parameters(init_fmaps=[8])
    param_set.add_parameters(zoom_level=[0])
    param_set.add_parameters(rseed=list(range(10)))
    param_set.add_parameters(loss=["bce"])
    param_set.add_parameters(sparse_start_after=[0, -1, 15, 50])
    param_set.add_parameters(aug_hed_bias_range=[0.025])
    param_set.add_parameters(aug_hed_coef_range=[0.025])
    param_set.add_parameters(aug_blur_sigma_extent=[0.1])
    param_set.add_parameters(aug_noise_var_extent=[0.05])
    param_set.add_parameters(lr_sched_factor=[0.5])
    param_set.add_parameters(lr_sched_patience=[5])
    param_set.add_parameters(lr_sched_cooldown=[10])
    param_set.add_parameters(save_cues=[False])
    param_set.add_parameters(sparse_data_rate=[1.0])
    param_set.add_parameters(sparse_data_max=[1.0])
    param_set.add_parameters(no_distillation=[False, True])
    param_set.add_parameters(no_groundtruth=[False])
    param_set.add_parameters(weights_mode=["pred_entropy", "pred_merged", "constant", "balance_gt", "pred_consistency"])
    param_set.add_parameters(weights_constant=[1.0, 0.5, 0.4])
    param_set.add_parameters(weights_consistency_fn=["quadratic", "absolute"])
    param_set.add_parameters(weights_minimum=[0.0, 0.5])
    param_set.add_parameters(weights_neighbourhood=[1, 2])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(exclude_no_new_data=exclude_no_new_data)
    constrained.add_constraints(exclude_no_groundtruth_no_pretraining_data=exclude_no_groundtruth_no_pretraining_data)
    constrained.add_constraints(at_least_one_source_of_annot=at_least_one_source_of_annot)
    constrained.add_constraints(weight_exclude=weight_exclude)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("thyroid-unet-training-weights", constrained, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)
