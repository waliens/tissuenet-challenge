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
    return not (kwargs.get("no_ground_truth") and kwargs.get("sparse_start_after") == -1)


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
    param_set.add_parameters(loss=["bce", "both", "dice"])
    param_set.add_parameters(sparse_start_after=[-1, 0, 10, 50])
    param_set.add_parameters(aug_hed_bias_range=[0.025])
    param_set.add_parameters(aug_hed_coef_range=[0.025])
    param_set.add_parameters(aug_blur_sigma_extent=[0.1])
    param_set.add_parameters(aug_noise_var_extent=[0.05])
    param_set.add_parameters(lr_sched_factor=[0.5])
    param_set.add_parameters(lr_sched_patience=[5])
    param_set.add_parameters(lr_sched_cooldown=[10])
    param_set.add_parameters(save_cues=[False])
    param_set.add_parameters(sparse_data_rate=[0.1, 0.5, 1.0])
    param_set.add_parameters(sparse_data_max=[1.0, -1])
    param_set.add_parameters(no_distillation=[True, False])
    param_set.add_parameters(no_groundtruth=[True, False])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(exclude_no_new_data=exclude_no_new_data)
    constrained.add_constraints(exclude_no_groundtruth_no_pretraining_data=exclude_no_groundtruth_no_pretraining_data)
    constrained.add_constraints(at_least_one_source_of_annot=at_least_one_source_of_annot)


    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("thyroid-unet-training-study", constrained, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)
