from clustertools import Computation


class ReevalMonusegComputation(Computation):
    def __init__(self, *args, n_jobs=1, data_path="./", model_path="./", **kwargs):
        super().__init__(*args, **kwargs)
        self._n_jobs = n_jobs
        self._data_path = data_path
        self._model_path = model_path

    def run(self, result, comp_index=None, **parameters):
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

        def get_hard_dice(y_true, y_pred, step=0.001):
            th_opt = Thresholdable(y_true, y_pred)
            thresholds, dices = thresh_exhaustive_eval(th_opt, eps=step)
            best_idx = np.argmax(dices)
            best_threshold, best_dice = thresholds[best_idx], dices[best_idx]
            return best_threshold, best_dice, thresholds, dices

        def progress(cls, start, end, perc):
            cls.notify_progress((start + perc * (end - start)) / 100)

        train_exp = "monuseg-unet-hard"
        comp_params, comp_results = load_computation(train_exp, comp_index)

        device = torch.device(device="cuda:0")
        unet = Unet(comp_params['init_fmaps'], n_classes=1)
        unet.to(device)

        def create_result_entries(r, entries, sets):
            for entry in entries:
                for _set in sets:
                    r["{}_{}".format(_set, entry)] = list()

        create_result_entries(
            result,
            ["hard_dice", "roc_auc", "cm",
             "avg_soft_dice", "avg_hard_dice", "avg_roc_auc", "avg_loss",
             "all_loss", "all_roc_auc", "all_hard_dice", "all_soft_dice"],
            ["val", "test"]
        )

        n_epochs = len(comp_results['save_path'])
        for epoch, model_filename in enumerate(comp_results['save_path']):
            progress_start = epoch / n_epochs
            progress_end = progress_start + (1 / n_epochs)
            progress(self, 0, 100, progress_start)
            print("Epoch '{}'".format(epoch))
            unet.load_state_dict(torch.load(os.path.join(self._model_path, model_filename)))
            unet.eval()

            n_sets = 2
            for set_idx, _set in enumerate(['validation', 'test']):
                _set_prefix = "val" if _set == "validation" else _set
                print("-> set '{}'".format(_set))
                folder_path = os.path.join(self._data_path, _set)
                image_path = os.path.join(folder_path, "images")
                mask_path = os.path.join(folder_path, "masks")
                crops = [MemoryCrop(
                    os.path.join(image_path, filename),
                    os.path.join(mask_path, filename.replace("tif", "png")),
                    tile_size=comp_params['tile_size']
                ) for filename in os.listdir(folder_path)]

                # scores
                losses = np.zeros(len(crops), dtype=np.float)
                roc_aucs = np.zeros(len(crops), dtype=np.float)
                soft_dices = np.zeros(len(crops), dtype=np.float)
                hard_dices = np.zeros(len(crops), dtype=np.float)
                dice_thresholds = np.zeros(len(crops), dtype=np.float)
                all_y_pred, all_y_true = np.array([]), np.array([])
                no_fg_counter = 0

                for i, crop in enumerate(crops):
                    progress(self, progress_start, progress_end, (set_idx / n_sets) + (i / len(crops)))
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

                result[_set_prefix + "_avg_loss"].append(avg_loss)
                result[_set_prefix + "_avg_roc_auc"].append(avg_roc_auc)
                result[_set_prefix + "_avg_hard_dice"].append(avg_hard_dice)
                result[_set_prefix + "_avg_soft_dice"].append(avg_soft_dice)
                result[_set_prefix + "_roc_auc"].append(roc_auc)
                result[_set_prefix + "_hard_dice"].append(hard_dice)
                result[_set_prefix + "_cm"].append(cm)
                result[_set_prefix + "_all_loss"].append(losses)
                result[_set_prefix + "_all_roc_auc"].append(roc_aucs)
                result[_set_prefix + "_all_hard_dice"].append(hard_dices)
                result[_set_prefix + "_all_soft_dice"].append(soft_dices)

