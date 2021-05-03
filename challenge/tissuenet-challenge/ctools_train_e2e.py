import os
from functools import partial

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
    return (arch in {"densenet121", "resnet50"} and (zlevel, bsize) in {(2, 32), (2, 24), (1, 8), (0, 2)}) or \
           (arch in {"resnet34", "resnet18"} and (zlevel, bsize) in {(2, 32), (2, 24), (1, 16), (0, 4)})


def cstrnt_pretraining(**kwargs):
    return kwargs["architecture"] in {"resnet50", "densenet121"} or kwargs["pretrained"] == "imagenet"


def cstrnt_low_lt_high(param_prefix="", **kwargs):
    return kwargs[param_prefix + "low"] < kwargs[param_prefix + "high"]


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
    param_set.add_parameters(architecture=["densenet121", "resnet34"])
    param_set.add_parameters(epochs=60)
    param_set.add_parameters(batch_size=[24])
    param_set.add_parameters(zoom_level=[2])
    param_set.add_parameters(train_size=0.8)
    param_set.add_parameters(random_seed=42)
    param_set.add_parameters(learning_rate=[0.001])
    param_set.add_parameters(aug_elastic_alpha_low=[80])
    param_set.add_parameters(aug_elastic_alpha_high=[120])
    param_set.add_parameters(aug_elastic_sigma_low=[9.0])
    param_set.add_parameters(aug_elastic_sigma_high=[11.0])
    param_set.add_parameters(aug_hed_bias_range=[0.0125, 0.025, 0.05, 0.1])
    param_set.add_parameters(aug_hed_coef_range=[0.0125, 0.025, 0.05, 0.1])

    param_set.add_separator()
    param_set.add_parameters(aug_elastic_alpha_high=[150])
    param_set.add_parameters(aug_elastic_sigma_low=[7.0])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(bsize_zoom_arch=cstrnt_batchsize_zoom)
    constrained.add_constraints(arch_pretr=cstrnt_pretraining)
    constrained.add_constraints(elastic_sigma=partial(cstrnt_low_lt_high, param_prefix="aug_elastic_sigma_"))
    constrained.add_constraints(elastic_sigma=partial(cstrnt_low_lt_high, param_prefix="aug_elastic_alpha_"))

    # Wrap it together as an experiment
    experiment = Experiment("tissuenet-e2e-train-3rd", constrained, CliComputationFactory(main, **env))

    # Finally run the experiment
    environment.run(experiment)