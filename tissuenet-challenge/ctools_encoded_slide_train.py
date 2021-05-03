import numpy as np

from clustertools import ParameterSet, ConstrainedParameterSet, Experiment, set_stdout_logging, CTParser

from cli_computation import CliComputationFactory
from encoded_slide_train import main


def env_parser():
    parser = CTParser()
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    return parser


if __name__ == "__main__":
    # Define the parameter set: the domain each variable can take
    set_stdout_logging()

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    data_path = "/scratch/users/rmormont/tissuenet"
    env = {"base_path": data_path,
           "image_path": "wsi_encoded",
           "metadata_path": "metadata",
           "model_path": "models/encoded",
           "device": namespace.device,
           "n_jobs": namespace.n_jobs
           }

    np.random.seed(42)

    param_set = ParameterSet()
    param_set.add_parameters(epochs=100)
    param_set.add_parameters(batch_size=[4])
    param_set.add_parameters(train_size=0.8)
    param_set.add_parameters(random_seed=np.random.randint(0, (1 << 32) - 1))
    param_set.add_parameters(learning_rate=[0.001])

    # Wrap it together as an experiment
    experiment = Experiment("tissuenet-encoded-train-2nd", param_set, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)