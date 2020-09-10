import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from wrapper import ProcessWSIComputation


def env_parser():
    parser = CTParser()
    parser.add_argument("--save_path", dest="save_path")
    parser.add_argument("--data_path", "--data_path", dest="data_path")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    parser.add_argument("--project_id", dest="project_id", default=-1, type=int)
    parser.add_argument("--software_id", dest="software_id", default=-1, type=int)
    parser.add_argument("--image_id", dest="image_id", default=-1, type=int)
    _ = Cytomine._add_cytomine_cli_args(parser.parser)
    return parser


if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()
    param_set.add_parameters(image_id=[77150767, 77150761, 77150809])
    param_set.add_parameters(batch_size=[8])
    param_set.add_parameters(tile_overlap=[0])
    param_set.add_parameters(tile_size=256)
    param_set.add_parameters(init_fmaps=8)
    param_set.add_parameters(zoom_level=2)
    param_set.add_separator()
    param_set.add_parameters(tile_size=512)
    param_set.add_parameters(zoom_level=0)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return ProcessWSIComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("thyroid-unet-inference", param_set, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)
