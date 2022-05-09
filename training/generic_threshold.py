import itertools

from clustertools import Computation
from clustertools.storage import PickleStorage


class TuneThresholdComputation(Computation):
    def __init__(self, exp_name, comp_name, host=None, private_key=None, public_key=None, n_jobs=1, device="cuda:0",
                 save_path=None, data_path=None, th_step=0.01, context="n/a",
                 storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._n_jobs = n_jobs
        self._device = device
        self._save_path = save_path
        self._cytomine_private_key = private_key
        self._cytomine_public_key = public_key
        self._cytomine_host = host
        self._data_path = data_path
        self._th_step = th_step

    def run(self, result, nbs="1", train_exp="none", comp_index=None, **parameters):
        import os
        import torch
        import numpy as np
        from unet import Unet
        from dataset import predict_set
        from augment import get_norm_transform
        from clustertools.experiment import load_computation
        from threshold_optimizer import Thresholdable, thresh_exhaustive_eval

        from cytomine import Cytomine

        from functools import partial
        from numpy.random import default_rng
        from monuseg import MonusegDatasetGenerator
        from segpc import SegpcDatasetGenerator
        from glas import GlaSDatasetGenerator

        # def progress(cls, start, end, _i, n):
        #     cls.notify_progress((start + (_i/n) * (end - start)))

        def create_result_entries(r, _metrics, _srcs):
            for _metric in _metrics:
                for _src in _srcs:
                    r["{}_{}".format(_src, _metric)] = -1

        def get_param_value(params, query, default=None):
            matches = [v for k, v in comp_params.items() if query in k]
            if len(matches) == 0:
                return default
            return matches[0]

        with Cytomine.connect(host=self._cytomine_host,
                              public_key=self._cytomine_public_key,
                              private_key=self._cytomine_private_key):
            nbs = list(map(int, nbs.split(",")))
            src_types = ["train", "cal", "test"]
            srcs = ["{}{}".format(src_type, nb) for src_type, nb in itertools.product(src_types, nbs)]
            metrics = ["dice_threshold", "hard_dice"]
            create_result_entries(result, metrics, srcs)

            comp_params, comp_results = load_computation(train_exp, comp_index)

            device = torch.device(device=self._device)
            unet = Unet(comp_params['init_fmaps'], n_classes=1)
            unet.to(device)
            model_filepath = os.path.join(self._save_path, comp_results['save_path'][-1])
            if not os.path.exists(model_filepath):
                raise ValueError("missing model '{}'".format(model_filepath))
            unet.load_state_dict(torch.load(model_filepath, map_location=device))
            unet.eval()

            base_args = [self._data_path, comp_params["tile_size"]]
            base_kwargs = {
                "missing_seed": get_param_value(comp_params, "_ms"),
                "n_complete": get_param_value(comp_params, "_nc"),
                "remove_ratio": get_param_value(comp_params, "_rr"),
                "n_validation": get_param_value(comp_params, "n_validation", default=0),
            }
            dataset_name = comp_params["dataset"]
            cls, args, kwargs = {
                "monuseg": (MonusegDatasetGenerator, base_args, base_kwargs),
                "segpc": (SegpcDatasetGenerator, base_args, base_kwargs),
                "glas": (GlaSDatasetGenerator, base_args, base_kwargs),
            }[dataset_name]

            dataset = cls(*args, **kwargs)
            incomplete_list, complete_list, test_list, validation_list = dataset.sets()

            def worker_init(*atgs, **kwargs):
                pass

            prgs_prct_end = 1.0 - len(nbs) / 10

            test_pred = predict_set(
                unet, test_list, device, in_trans=get_norm_transform(),
                overlap=comp_params["overlap"], batch_size=comp_params["batch_size"],
                n_jobs=self._n_jobs, worker_init_fn=worker_init,
                progress_fn=None) #partial(progress, self, 0, prgs_prct_end))

            test_true = [np.asarray(crop.crop_and_mask()[1]).astype(np.uint8) / 255 for crop in test_list]
            all_y_true = np.hstack([v.flatten() for v in test_true])
            all_y_pred = np.hstack([v.flatten() for v in test_pred])
            all_which = np.hstack([np.full(np.prod(img.shape), i) for i, img in enumerate(test_pred)])
            imgs_indexes = np.arange(len(test_pred))

            rseed = base_kwargs["missing_seed"]
            if rseed == 42:
                rseed = comp_params["rseed"]

            generator = default_rng(rseed)
            train_thresh = comp_results["cal_threshold"][-1]

            for i, nb_cal in enumerate(nbs):
                cal_set = generator.choice(imgs_indexes, nb_cal, replace=False)
                eval_set = np.setdiff1d(imgs_indexes, cal_set)
                cal_set_idxs = np.in1d(all_which, cal_set)

                cal_th_opt = Thresholdable(all_y_true[cal_set_idxs], all_y_pred[cal_set_idxs])
                cal_thresholds, cal_dices = thresh_exhaustive_eval(cal_th_opt, eps=self._th_step)
                cal_best_idx = np.argmax(cal_dices)
                cal_thresh, oncal_dice = cal_thresholds[cal_best_idx], cal_dices[cal_best_idx]

                eval_set_idxs = np.in1d(all_which, eval_set)
                test_th_opt = Thresholdable(all_y_true[eval_set_idxs], all_y_pred[eval_set_idxs])
                test_thresholds, test_dices = thresh_exhaustive_eval(test_th_opt, eps=self._th_step)
                best_idx = np.argmax(test_dices)
                test_thresh, test_dice = test_thresholds[best_idx], test_dices[best_idx]

                cal_dice = test_th_opt.eval(cal_thresh)
                train_dice = test_th_opt.eval(train_thresh)

                result["oncal{}_hard_dice".format(nb_cal)] = oncal_dice
                result["train{}_dice_threshold".format(nb_cal)] = train_thresh
                result["train{}_hard_dice".format(nb_cal)] = train_dice
                result["cal{}_dice_threshold".format(nb_cal)] = cal_thresh
                result["cal{}_hard_dice".format(nb_cal)] = cal_dice
                result["test{}_dice_threshold".format(nb_cal)] = test_thresh
                result["test{}_hard_dice".format(nb_cal)] = test_dice

                print(train_dice, oncal_dice, test_dice)
                print(train_thresh, cal_thresh, test_thresh)
                #progress(self, prgs_prct_end, 1.0, i, len(nbs))
