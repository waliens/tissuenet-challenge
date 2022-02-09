import os
from clustertools import set_stdout_logging, ParameterSet, Experiment, CTParser, ConstrainedParameterSet
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from generic_train import TrainComputation


def env_parser():
    parser = CTParser()
    parser.add_argument("--save_path", dest="save_path")
    parser.add_argument("--data_path", "--data_path", dest="data_path")
    parser.add_argument("--device", dest="device", default="cuda:0")
    parser.add_argument("--n_jobs", dest="n_jobs", default=1, type=int)
    parser.add_argument("--th_step", dest="th_step", default=0.01, type=float)
    _ = Cytomine._add_cytomine_cli_args(parser.parser)
    return parser


def weight_exclude(**kwargs):
    wconstant_is_one = 0.99 < kwargs["weights_constant"] < 1.01
    wmode_is_not_constant = kwargs["weights_mode"] != "constant"
    wmode_is_constant = not wmode_is_not_constant
    min_weight_is_zero = kwargs.get("weights_minimum") < 0.01
    constant_and_is_one = (wmode_is_constant and wconstant_is_one)
    # in line order:
    # > wmode=constant => constant = 1
    # > constant must be one for other experiments wmode=constant
    # > if wmode is not  consistency fn=quad & nh=2
    # > wmin=0 if wmode=const or balanced
    # > wmin>0 if wmode includes entropy
    return (wmode_is_constant or wconstant_is_one) \
       and (kwargs.get("weights_mode") in {"pred_consistency", "pred_merged"} or
            (kwargs.get("weights_consistency_fn") == "quadratic" and kwargs.get("weights_neighbourhood") == 2)) \
       and (kwargs.get("weights_mode") not in {"constant", "balance_gt"} or min_weight_is_zero) \
       and (kwargs.get("weights_mode") not in {"pred_entropy", "pred_merged"} or not min_weight_is_zero)


def exclude_target_and_dice_calibration(**kwargs):
    target_mode = kwargs["distil_target_mode"]
    n_calibration = kwargs["n_calibration"]
    n_complete = kwargs["monu_nc"]
    cond = (target_mode == "soft" or n_calibration > 0) and n_complete > n_calibration
    return cond


def no_distillation_filter(**kwargs):
    no_distillation = kwargs.get("no_distillation")
    cond = not no_distillation or (
        kwargs.get("weights_mode") == "constant"
        and 0.99 < kwargs.get("weights_constant") < 1.01
        and kwargs.get("weights_consistency_fn") == "quadratic"
        and kwargs.get("weights_minimum") < 0.01
        and kwargs.get("weights_neighbourhood") == 2
        and kwargs.get("distil_target_mode") == "soft"
    )
    return cond


# def wmode_exclude_no_distil(**kwargs):
#     return kwargs.get("weights_mode") not in {"pred_consistency", "pred_merged", "pred_entropy"} or (
#         kwargs.get("sparse_start_after") < 50
#         and not kwargs.get("no_distillation")
#     )


# def not_using_incomplete(**kwargs):
#     return kwargs.get("sparse_start_after") != 50 or (
#             kwargs.get("weights_mode") == "constant"
#             and kwargs.get("weights_constant") > 0.99
#             and kwargs.get("weights_consistency_fn") == "quadratic"
#             and kwargs.get("weights_minimum") < 0.01
#             and kwargs.get("weights_neighbourhood") == 2
#             and kwargs.get("no_distillation")
#             and not kwargs.get("no_groundtruth")
#             and kwargs.get("sparse_data_rate") > 0.99
#             and kwargs.get("sparse_data_max") > 0.99)


if __name__ == "__main__":
    set_stdout_logging()
    # Define the parameter set: the domain each variable can take

    environment, namespace = env_parser().parse()
    env_params = dict(namespace._get_kwargs())
    os.makedirs(namespace.save_path, exist_ok=True)

    param_set = ParameterSet()

    param_set.add_parameters(dataset="monuseg")
    param_set.add_parameters(monu_ms=[13315092, 21081788, 26735830, 35788921, 56755036, 56882282, 65682867, 91090292, 93410762, 96319575])
    param_set.add_parameters(monu_rr=[0.9])
    param_set.add_parameters(monu_nc=[1, 2])
    param_set.add_parameters(iter_per_epoch=150)
    param_set.add_parameters(batch_size=8)
    param_set.add_parameters(epochs=50)
    param_set.add_parameters(overlap=0)
    param_set.add_parameters(tile_size=512)
    param_set.add_parameters(lr=0.001)
    param_set.add_parameters(init_fmaps=8)
    param_set.add_parameters(zoom_level=0)
    param_set.add_parameters(rseed=42)
    param_set.add_parameters(loss="bce")
    param_set.add_parameters(aug_hed_bias_range=0.025)
    param_set.add_parameters(aug_hed_coef_range=0.025)
    param_set.add_parameters(aug_blur_sigma_extent=0.1)
    param_set.add_parameters(aug_noise_var_extent=0.05)
    param_set.add_parameters(lr_sched_factor=0.5)
    param_set.add_parameters(lr_sched_patience=5)
    param_set.add_parameters(lr_sched_cooldown=10)
    param_set.add_parameters(save_cues=False)
    param_set.add_parameters(sparse_data_rate=1.0)
    param_set.add_parameters(sparse_data_max=1.0)
    param_set.add_parameters(sparse_start_after=[15])
    param_set.add_parameters(no_distillation=[False, True])
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode=["constant", "balance_gt", "pred_consistency", "pred_entropy", "pred_merged"])
    param_set.add_parameters(weights_constant=[1.0])
    param_set.add_parameters(weights_consistency_fn=["quadratic"])
    param_set.add_parameters(weights_minimum=[0.0, 0.5])
    param_set.add_parameters(weights_neighbourhood=[1, 2])
    param_set.add_parameters(distil_target_mode=["soft", "hard_dice"])
    param_set.add_parameters(n_calibration=[0, 1])

    # param_set.add_separator()
    # param_set.add_parameters(monu_rr=[0.25, 0.5, 0.75])
    # param_set.add_parameters(monu_nc=[3, 4, 5])
    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)
    constrained.add_constraints(exclude_target_and_dice_calibration=exclude_target_and_dice_calibration)
    constrained.add_constraints(no_distillation=no_distillation_filter)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn


    # Wrap it together as an experiment
    experiment = Experiment("monuseg-unet-hard", constrained, make_build_fn(**env_params))

    # Finally run the experiment
    environment.run(experiment)

"""
 
~/miniconda3/envs/py-gpu/bin/python train_monuseg_hard_clustertools.py --save_path /home/rmormont/models/monuseg-unet/hard/weights --data_path /scratch/users/rmormont/monuseg --host="https://research.cytomine.be" --public_key="13cad665-8b7a-4f9f-a6c0-93166e33ddb2" --private_key="520c4556-7f36-4cf1-82af-010f6be9adb3"  --device "cuda:0" --n_jobs 4 --th_step 0.0005 debug
 
~/miniconda3/envs/py-gpu/bin/python train_monuseg_hard_clustertools.py --save_path /home/rmormont/models/monuseg-unet/hard/weights --data_path /scratch/users/rmormont/monuseg --host="https://research.cytomine.be" --public_key="13cad665-8b7a-4f9f-a6c0-93166e33ddb2" --private_key="520c4556-7f36-4cf1-82af-010f6be9adb3" --device "cuda:0" --n_jobs 4 --th_step 0.0005 slurm --n_proc 4 --gpu 1 --memory 8G --time 10:00:00
"""
