import os
from clustertools import ParameterSet, ConstrainedParameterSet, Experiment, set_stdout_logging, CTParser

from cli_computation import CliComputationFactory
from e2e_classifier_train import main


def env_parser():
    parser = CTParser()
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    return parser


def cstrnt_batchsize_zoom(**kwargs):
    zlevel = kwargs["zoom_level"]
    bsize = kwargs["batch_size"]
    arch = kwargs["architecture"]
    return (arch in {"densenet121", "resnet50"} and (zlevel, bsize) in {(3, 32), (2, 32), (1, 8), (0, 2)}) or \
           (arch in {"resnet34", "resnet18"} and (zlevel, bsize) in {(3, 32), (2, 32), (1, 16), (0, 4)})


def cstrnt_pretraining(**kwargs):
    return kwargs["architecture"] in {"resnet50", "densenet121"} or kwargs["pretrained"] == "imagenet"


if __name__ == "__main__":
    # Define the parameter set: the domain each variable can take
    set_stdout_logging()

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
    param_set.add_parameters(pretrained=["imagenet", "mtdp"])
    param_set.add_parameters(architecture=["densenet121", "resnet50", "resnet18", "resnet34"])
    param_set.add_parameters(epochs=60)
    param_set.add_parameters(batch_size=[32, 16, 8])
    param_set.add_parameters(zoom_level=[2, 1])
    param_set.add_parameters(train_size=0.8)
    param_set.add_parameters(random_seed=42)
    param_set.add_parameters(learning_rate=[0.001, 0.0001])
    param_set.add_separator()
    param_set.add_parameters(zoom_level=[3])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(bsize_zoom_arch=cstrnt_batchsize_zoom)
    constrained.add_constraints(arch_pretr=cstrnt_pretraining)

    # Wrap it together as an experiment
    experiment = Experiment("tissuenet-e2e-train-2nd", constrained, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)