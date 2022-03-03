import os

from clustertools import Computation
from clustertools.storage import PickleStorage


class ReevalMonusegComputation(Computation):
    def __init__(self, exp_name, comp_name, n_jobs=1, device="cpu", data_path="./", model_path="./",
                 context="n/a", storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._device = device
        self._n_jobs = n_jobs
        self._data_path = data_path
        self._model_path = model_path

    def run(self, result, sets="val,test", train_exp="none", comp_index=None, **parameters):
        import os
        import torch
        import numpy as np
        from unet import Unet
        from sklearn import metrics
        from augment import get_norm_transform
        from dataset import MemoryCrop, predict_roi
        from clustertools.experiment import load_computation
        from threshold_optimizer import Thresholdable, thresh_exhaustive_eval
        from generic_train import soft_dice_coefficient

        def get_hard_dice(_y_true, _y_pred, step=0.001):
            th_opt = Thresholdable(_y_true, _y_pred)
            _thresholds, dices = thresh_exhaustive_eval(th_opt, eps=step)
            best_idx = np.argmax(dices)
            best_threshold, best_dice = _thresholds[best_idx], dices[best_idx]
            return best_threshold, best_dice, _thresholds, dices

        def progress(cls, start, end, _i, n):
            cls.notify_progress((start + (_i/n) * (end - start)))

        def fill_for_missing_epoch(res, _sets):
            for __set in _sets:
                res[__set + "_avg_loss"].append(-1)
                res[__set + "_avg_roc_auc"].append(-1)
                res[__set + "_avg_hard_dice"].append(-1)
                res[__set + "_avg_soft_dice"].append(-1)
                res[__set + "_roc_auc"].append(-1)
                res[__set + "_hard_dice"].append(-1)
                res[__set + "_cm"].append(np.array([]))
                res[__set + "_all_loss"].append(np.array([]))
                res[__set + "_all_roc_auc"].append(np.array([]))
                res[__set + "_all_hard_dice"].append(np.array([]))
                res[__set + "_all_soft_dice"].append(np.array([]))

        comp_params, comp_results = load_computation(train_exp, comp_index)

        device = torch.device(device="cuda:0")
        unet = Unet(comp_params['init_fmaps'], n_classes=1)
        unet.to(device)

        def create_result_entries(r, entries, _sets):
            for entry in entries:
                for __set in _sets:
                    r["{}_{}".format(__set, entry)] = list()

        create_result_entries(
            result,
            ["hard_dice", "roc_auc", "cm",
             "avg_soft_dice", "avg_hard_dice", "avg_roc_auc", "avg_loss",
             "all_loss", "all_roc_auc", "all_hard_dice", "all_soft_dice"],
            ["val", "test"]
        )

        sets = sets.split(",")
        n_epochs = len(comp_results['save_path'])
        for epoch, model_filename in enumerate(comp_results['save_path']):
            progress_start = epoch / n_epochs
            progress_end = progress_start + (1 / n_epochs)
            progress(self, 0, 1, epoch, n_epochs)
            print("Epoch '{}'".format(epoch))
            model_filepath = os.path.join(self._model_path, model_filename)
            if os.path.exists(model_filepath):
                unet.load_state_dict(torch.load(model_filepath))
                unet.eval()
                print("/!\\ model '{}' missing".format(model_filepath))
            else:
                fill_for_missing_epoch(result, sets)
                continue

            n_sets = len(sets)
            for set_idx, _set in enumerate(sets):
                print("-> set '{}'".format(_set))
                folder_path = os.path.join(self._data_path, _set)
                image_path = os.path.join(folder_path, "images")
                mask_path = os.path.join(folder_path, "masks")
                crops = [MemoryCrop(
                    os.path.join(image_path, filename),
                    os.path.join(mask_path, filename.replace("tif", "png")),
                    tile_size=comp_params['tile_size']
                ) for filename in os.listdir(image_path)]

                # scores
                losses = np.zeros(len(crops), dtype=np.float)
                roc_aucs = np.zeros(len(crops), dtype=np.float)
                soft_dices = np.zeros(len(crops), dtype=np.float)
                hard_dices = np.zeros(len(crops), dtype=np.float)
                dice_thresholds = np.zeros(len(crops), dtype=np.float)
                all_y_pred, all_y_true = np.array([]), np.array([])
                no_fg_counter = 0

                for i, crop in enumerate(crops):
                    progress(self, progress_start, progress_end, len(crops) * set_idx + i, n_sets * len(crops))
                    print("---> {}/{}".format(i + 1, len(crops)))
                    with torch.no_grad():
                        y_pred, _ = predict_roi(
                            crop, [], unet, device,
                            in_trans=get_norm_transform(),
                            batch_size=comp_params['batch_size'],
                            tile_size=comp_params['tile_size'],
                            overlap=comp_params['overlap'],
                            n_jobs=self._n_jobs,
                            zoom_level=comp_params['zoom_level']
                        )

                    _, y_true, _, _, _ = crop.crop_and_mask()
                    y_true = np.asarray(y_true)

                    all_y_pred = np.hstack([all_y_pred, y_pred.flatten()])
                    all_y_true = np.hstack([all_y_true, y_true.flatten()])

                    losses[i] = metrics.log_loss(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
                    if np.count_nonzero(y_true) == 0:
                        no_fg_counter += 1
                        roc_aucs[i] = 0
                        dice_thresholds[i], hard_dices[i] = -1, 0
                    else:
                        roc_aucs[i] = metrics.roc_auc_score(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
                        dice_thresholds[i], hard_dices[i], _, _ = get_hard_dice(y_true.flatten(), y_pred.flatten())
                    soft_dices[i] = soft_dice_coefficient(y_true, y_pred)

                avg_loss = np.mean(losses)
                corrective = len(crops) / (len(crops) - no_fg_counter)
                avg_roc_auc = np.mean(roc_aucs) * corrective
                avg_hard_dice = np.mean(hard_dices) * corrective
                avg_soft_dice = np.mean(soft_dices)

                dice_threshold, hard_dice, thresholds, hard_dices = get_hard_dice(all_y_true, all_y_pred)
                roc_auc = metrics.roc_auc_score(all_y_true, all_y_pred, labels=[0, 1])
                cm = metrics.confusion_matrix(all_y_true.astype(np.uint8), (all_y_pred.flatten() > dice_threshold).astype(np.uint8))
                del all_y_pred, all_y_true

                result[_set + "_avg_loss"].append(avg_loss)
                result[_set + "_avg_roc_auc"].append(avg_roc_auc)
                result[_set + "_avg_hard_dice"].append(avg_hard_dice)
                result[_set + "_avg_soft_dice"].append(avg_soft_dice)
                result[_set + "_roc_auc"].append(roc_auc)
                result[_set + "_hard_dice"].append(hard_dice)
                result[_set + "_cm"].append(cm)
                result[_set + "_all_loss"].append(losses)
                result[_set + "_all_roc_auc"].append(roc_aucs)
                result[_set + "_all_hard_dice"].append(hard_dices)
                result[_set + "_all_soft_dice"].append(soft_dices)

        return result

