import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, ConstrainedParameterSet
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from apply_model_over_epochs import ApplyModelComputation


def env_parser():
    parser = CTParser()
    parser.add_argument("--save_path", dest="save_path")
    parser.add_argument("--data_path", dest="data_path")
    parser.add_argument("--model_dir", dest="model_dir")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    parser.add_argument("--th_step", dest="th_step", default=0.01, type=float)
    _ = Cytomine._add_cytomine_cli_args(parser.parser)
    return parser


if __name__ == "__main__":
    """
    python apply_model_clustertools.py  --model_dir /home/rmormont/models/monuseg-unet/missing --save_path /home/rmormont/images/monuseg --data_path /scratch/users/rmormont/monuseg  --host="https://research.cytomine.be"  --public_key="13cad665-8b7a-4f9f-a6c0-93166e33ddb2"  --private_key="520c4556-7f36-4cf1-82af-010f6be9adb3"  --device "cuda:0"  --n_jobs 4  slurm  --n_proc 4  --gpu 1  --memory 4G  --time 5:00 --partition debug --capacity 1
    """
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take
    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()

    param_set.add_parameters(batch_size="8")
    param_set.add_parameters(tile_size=512)
    param_set.add_parameters(zoom_level=0)
    param_set.add_parameters(dataset="monuseg")
    param_set.add_parameters(epoch=list(range(50)))
    param_set.add_parameters(experiment="monuseg-unet-missing")
    param_set.add_parameters(exp_rr=0.9)
    param_set.add_parameters(exp_ms=13315092)
    param_set.add_parameters(exp_nc=[1])
    param_set.add_parameters(exp_weights_mode="constant")
    param_set.add_parameters(exp_sparse_start_after=[-1, 0, 15])
    param_set.add_parameters(image="TCGA-AO-A0J2-01A-01-BSA.tif")
    param_set.add_separator()
    param_set.add_parameters(exp_rr=[0.25, 0.5, 0.75])
    param_set.add_separator()
    param_set.add_parameters(image="TCGA-AC-A2FO-01A-01-TS1.tif")

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return ApplyModelComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)
        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("apply-model-monuseg", param_set, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)
