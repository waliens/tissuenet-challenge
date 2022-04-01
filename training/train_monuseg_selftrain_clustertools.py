import itertools
import os
from collections import defaultdict

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
       and (kwargs.get("weights_mode") not in {"constant", "balance_gt", "pred_consistency"} or min_weight_is_zero) \
       and (kwargs.get("weights_mode") not in {"pred_entropy", "pred_merged"} or not min_weight_is_zero)


from prettytable import PrettyTable as pt


def float2str(v):
    if isinstance(v, float):
        return "{:1.4f}".format(v)
    return v


def computation_changing_parameters(exp: Experiment, env, excluded=None):
    if excluded is None:
        excluded = set()
    else:
        excluded = set(excluded)
    computations = list(exp.yield_computations(env.context()))
    parameters = defaultdict(set)
    for comp in computations:
        for param, value in comp.parameters.items():
            if param not in excluded:
                parameters[param].add(float2str(value))
                parameters[param].add(float2str(value))

    changing_parameters = [pname for pname, value_set in parameters.items() if len(value_set) > 1]
    processed = set()

    pre_excluded = list(excluded)
    tb = pt()
    # Add headers
    tb.field_names = ["ID"] + [p for p in pre_excluded] + changing_parameters
    # Add rows
    for comp in computations:
        param_comb_id = tuple(float2str(comp.parameters[pname]) for pname in changing_parameters)
        if param_comb_id in excluded:
            continue
        excluded.add(param_comb_id)
        row = [int(comp.comp_name.rsplit("-", 1)[-1])]
        for excl_param in pre_excluded:
            row.append(comp.parameters[excl_param])
        for param in changing_parameters:
            row.append(comp.parameters[param])
        tb.add_row(row)

    print(tb)


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
    param_set.add_parameters(monu_nc=[2])
    param_set.add_parameters(iter_per_epoch=100)
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
    param_set.add_parameters(sparse_start_after=10)
    param_set.add_parameters(no_distillation=False)
    param_set.add_parameters(no_groundtruth=False)
    param_set.add_parameters(weights_mode=["constant", "balance_gt", "pred_consistency", "pred_entropy", "pred_merged"])
    param_set.add_parameters(weights_constant=[1.0, 0.5, 0.25, 0.1, 0.05, 0.01])
    param_set.add_parameters(weights_consistency_fn=["quadratic", "absolute"])
    param_set.add_parameters(weights_minimum=[0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.0])
    param_set.add_parameters(weights_neighbourhood=[1, 2])
    param_set.add_parameters(distil_target_mode=["soft", "hard_dice"])

    constrained = ConstrainedParameterSet(param_set)
    constrained.add_constraints(weight_exclude=weight_exclude)

    def make_build_fn(**kwargs):
        def build_fn(exp_name, comp_name, context="n/a", storage_factory=PickleStorage):
            return TrainComputation(exp_name, comp_name, **kwargs, context=context, storage_factory=storage_factory)

        return build_fn

    # Wrap it together as an experiment
    experiment = Experiment("monuseg-unet-selftrain", constrained, make_build_fn(**env_params))

    computation_changing_parameters(experiment, environment, excluded={"monu_ms"})

    # Finally run the experiment
    environment.run(experiment)

