import itertools
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
    datacube = build_datacube("tissuenet-e2e-train-3rd")

    # zoom, arch, pretraining, lr

    param_set = ExplicitParameterSet()

    base_params = ["architecture", "pretrained"]
    aug_params = list(set(datacube.parameters).difference(base_params))
    print(aug_params)
    first_part = datacube(architecture="densenet121", pretrained="mtdp").iter_dimensions(*base_params)
    second_part = datacube(architecture="densenet121", pretrained="imagenet").iter_dimensions(*base_params)
    third_part = datacube(architecture="resnet34", pretrained="imagenet").iter_dimensions(*base_params)
    for (arch, pretrained), cube in itertools.chain(first_part, second_part, third_part):
        for aug_pvalues, aug_cube in cube.iter_dimensions(*aug_params):
            if aug_cube("val_acc") is None:
                continue

            filename = aug_cube("models")[-1]

            print(arch, pretrained, " ".join(aug_pvalues), filename, aug_cube["val_acc"][-1])
            # print("\"{}\": {},".format(filename, lr))

            param_set.add_parameter_tuple(
                architecture=arch,
                zoom_level=cube.metadata["zoom_level"],
                model_filename=filename,
                tile_size=320,
                train_size=0.8,
                batch_size=32,
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
    experiment = Experiment("tissuenet-e2e-eval-3rd", param_set, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)