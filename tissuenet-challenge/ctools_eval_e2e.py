import os
import numpy as np
from clustertools import ParameterSet, ConstrainedParameterSet, Experiment, set_stdout_logging, CTParser, \
    build_datacube, ExplicitParameterSet

from cli_computation import CliComputationFactory
from e2e_classifier_eval import main


def env_parser():
    parser = CTParser()
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    return parser


def cstrnt_pretraining(**kwargs):
    return kwargs["architecture"] in {"resnet50", "densenet121"} or kwargs["pretrained"] == "imagenet"


if __name__ == "__main__":
    # Define the parameter set: the domain each variable can take
    datacube = build_datacube("tissuenet-e2e-train-2nd")

    # zoom, arch, pretraining, lr

    param_set = ExplicitParameterSet()

    for (zoom, arch, pretrained, lr), cube in datacube.iter_dimensions("zoom_level", "architecture", "pretrained", "learning_rate"):

        for (batch_size,), batch_size_cube in cube.iter_dimensions("batch_size"):
            if batch_size_cube("val_acc") is None:
                continue

            best_epoch = np.argmax(batch_size_cube("val_acc"))

            filename = batch_size_cube("models")[best_epoch]

            # print("best", best_epoch, filename, batch_size_cube["val_acc"][best_epoch])
            # print("\"{}\": {},".format(filename, lr))

            param_set.add_parameter_tuple(
                architecture=arch,
                zoom_level=int(zoom),
                model_filename=filename,
                tile_size={1: 640, 2: 320, 3: 320}[int(zoom)],
                train_size=0.8,
                batch_size={1: 8, 2: 32,  3: 32}[int(zoom)],
                random_seed=42
            )

    set_stdout_logging()

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())

    data_path = "/scratch/users/rmormont/tissuenet"
    env = {"image_path": os.path.join(data_path, "wsis"),
           "metadata_path": os.path.join(data_path, "metadata"),
           "model_path": os.path.join(data_path, "models"),
           "device": namespace.device,
           "n_jobs": namespace.n_jobs
           }

    os.makedirs(env["image_path"], exist_ok=True)
    os.makedirs(env["metadata_path"], exist_ok=True)
    os.makedirs(env["model_path"], exist_ok=True)

    print(environment)
    # Wrap it together as an experiment
    experiment = Experiment("tissuenet-e2e-eval-2nd", param_set, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)