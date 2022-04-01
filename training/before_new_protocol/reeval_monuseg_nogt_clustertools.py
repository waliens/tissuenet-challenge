import os
import pickle

from train_monuseg_hard_clustertools import weight_exclude, exclude_target_and_dice_calibration, no_distillation_filter, \
    filter_nc_rr, min_weight_only_for_entropy
from clustertools import set_stdout_logging, Experiment, ParameterSet, CTParser, build_datacube
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

    exp_name = "monuseg-unet-nogt"

    setattr(main, "weight_exclude", weight_exclude)
    setattr(main, "exclude_target_and_dice_calibration", exclude_target_and_dice_calibration)
    setattr(main, "no_distillation_filter", no_distillation_filter)
    setattr(main, "filter_nc_rr", filter_nc_rr)
    setattr(main, "min_weight_only_for_entropy", min_weight_only_for_entropy)

    results_path = os.path.join("/home", "rmormont", "clustertools_data", "exp_monuseg-unet-nogt", "results")
    indexes = sorted([int(filename.split(".")[0].split("-")[-1]) for filename in os.listdir(results_path)])

    param_set.add_parameters(sets="test,val")
    param_set.add_parameters(train_exp=exp_name)
    for comp_index in indexes:
        param_set.add_parameters(comp_index=comp_index)
        param_set.add_separator()

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return ReevalMonusegComputation(exp_name, comp_name, **kwargs, context=context,
                                            storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-reeval-nogt", param_set, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment)

    # Finally run the experimenthis
    environment.run(experiment)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
