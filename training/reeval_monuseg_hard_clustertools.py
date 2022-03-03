import pickle

from train_monuseg_hard_clustertools import weight_exclude, exclude_target_and_dice_calibration, no_distillation_filter, \
    filter_nc_rr, min_weight_only_for_entropy
from clustertools import set_stdout_logging, Experiment, ParameterSet, CTParser
from clustertools.parameterset import build_parameter_set
from clustertools.storage import PickleStorage

from generic_reeval import ReevalMonusegComputation
from train_segpc_hard_clustertools import computation_changing_parameters


def env_parser():
    parser = CTParser()
    parser.add_argument("--model_path", dest="model_path")
    parser.add_argument("--data_path", dest="data_path")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    return parser


def main(argv):
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    param_set = ParameterSet()

    exp_name = "monuseg-unet-hard"

    setattr(main, "weight_exclude", weight_exclude)
    setattr(main, "exclude_target_and_dice_calibration", exclude_target_and_dice_calibration)
    setattr(main, "no_distillation_filter", no_distillation_filter)
    setattr(main, "filter_nc_rr", filter_nc_rr)
    setattr(main, "min_weight_only_for_entropy", min_weight_only_for_entropy)

    source_param_set = build_parameter_set(exp_name)
    indexes, _ = zip(*list(source_param_set))

    param_set.add_parameters(sets="test,val")
    param_set.add_parameters(train_exp=exp_name)
    param_set.add_parameters(comp_index=indexes)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return ReevalMonusegComputation(exp_name, comp_name, **kwargs, context=context,
                                            storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-reeval-hard", param_set, make_build_fn(**env_params))

    # computation_changing_parameters(experiment, environment)

    # Finally run the experiment
    environment.run(experiment)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
