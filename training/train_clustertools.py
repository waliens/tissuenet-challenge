import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from train import TrainComputation


def env_parser():
    parser = CTParser()
    parser.add_argument("--save_path", dest="save_path")
    parser.add_argument("--data_path", "--data_path", dest="data_path")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    _ = Cytomine._add_cytomine_cli_args(parser.parser)
    return parser


if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    param_set.add_parameters(batch_size=[8])
    param_set.add_parameters(epochs=[5])
    param_set.add_parameters(overlap=[0])
    param_set.add_parameters(tile_size=[512])
    param_set.add_parameters(lr=[0.001])
    param_set.add_parameters(init_fmaps=[8])

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("thyroid-unet-training", param_set, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)
