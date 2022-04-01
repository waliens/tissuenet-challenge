import os

from clustertools import set_stdout_logging, Experiment, ParameterSet, CTParser
from clustertools.storage import PickleStorage

from generic_reeval import ReevalMonusegComputation


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

    exp_name = "monuseg-unet-lset"

    results_path = os.path.join("/home", "rmormont", "clustertools_data", "exp_monuseg-unet-lset", "results")
    indexes = sorted([int(filename.split(".")[0].split("-")[-1]) for filename in os.listdir(results_path)])

    param_set.add_parameters(sets="test,val")
    param_set.add_parameters(train_exp=exp_name)
    for comp_index in indexes:
        param_set.add_parameters(comp_index=comp_index)
        if comp_index != indexes[-1]:
            param_set.add_separator()

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return ReevalMonusegComputation(exp_name, comp_name, **kwargs, context=context,
                                            storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-reeval-lset", param_set, make_build_fn(**env_params))

    # Finally run the experimenthis
    environment.run(experiment)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
