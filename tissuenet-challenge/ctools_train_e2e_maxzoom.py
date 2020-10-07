import os
from clustertools import ParameterSet, ConstrainedParameterSet, Experiment, set_stdout_logging, CTParser

from cli_computation import CliComputationFactory
from e2e_classifier_train_maxzoom import main


def env_parser():
    parser = CTParser()
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    return parser


if __name__ == "__main__":

    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    data_path = "/scratch/users/rmormont/tissuenet"
    env = {"image_path": os.path.join(data_path, "patches"),
           "metadata_path": os.path.join(data_path, "metadata"),
           "model_path": os.path.join(data_path, "models"),
           "device": namespace.device,
           "n_jobs": namespace.n_jobs
           }

    os.makedirs(env["image_path"], exist_ok=True)
    os.makedirs(env["metadata_path"], exist_ok=True)
    os.makedirs(env["model_path"], exist_ok=True)

    param_set = ParameterSet()
    param_set.add_parameters(epochs=40)
    param_set.add_parameters(batch_size=[8])
    param_set.add_parameters(zoom_level=[0])
    param_set.add_parameters(train_size=0.7)
    param_set.add_parameters(random_seed=42)
    param_set.add_parameters(learning_rate=[0.001])

    # Wrap it together as an experiment
    experiment = Experiment("tissuenet-e2e-train-maxzoom", param_set, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)