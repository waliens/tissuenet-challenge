import os
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import numpy as np

from clustertools import Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from sklearn import metrics
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from torchvision.transforms import transforms

from augment import get_aug_transforms, get_norm_transform
from dataset import CropTrainDataset, predict_roi, predict_annotation_crops_with_cues, GraduallyAddMoreDataState, \
    CropWithThresholdedCue
from segpc import SegpcDatasetGenerator
from threshold_optimizer import Thresholdable, thresh_exhaustive_eval
from monuseg import MonusegDatasetGenerator
from thyroid import ThyroidDatasetGenerator
from unet import Unet, DiceWithLogitsLoss, MergedLoss, BCEWithWeights
from weight_generator import WeightComputer
from skimage import io


def vstack(*args):
    if len(args) == 0:
        return
    out = args[0]
    i = 1
    while out.ndim == 1 and out.shape[0] == 0:
        out = args[i]
        i += 1
    return np.vstack([out, *args[i:]])



def get_random_init_fn(seed):
    def random_init():
        import numpy as np
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed + 42)
        np.random.seed(seed + 83)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return random_init


def soft_dice_coefficient(y_true, y_pred, epsilon=1e-6):
    '''
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    # skip the batch and class axis for calculating Dice score
    numerator = 2. * np.sum(y_pred * y_true)
    denominator = np.sum(np.square(y_pred) + np.square(y_true))
    return np.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch


def torange0_1(t):
    return t / 255.0


def worker_init(tid):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def save_cues(path, crops_with_cues):
    os.makedirs(path, exist_ok=True)
    for crop in crops_with_cues:
        io.imsave(os.path.join(path, crop.crop.annotation.originalFilename), crop.cue)


def determine_optimal_threshold(calibration_list, dataset, model, args, device):
    all_y_pred, all_y_true = np.array([]), np.array([])
    for calibration_roi in calibration_list:
        y_pred, y_true = predict_roi(
            calibration_roi,
            dataset.val_roi_foreground(calibration_roi),
            model, device,
            in_trans=get_norm_transform(),
            batch_size=args.batch_size,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            n_jobs=args.n_jobs,
            zoom_level=args.zoom_level
        )
        all_y_pred = np.hstack([all_y_pred, y_pred.flatten()])
        all_y_true = np.hstack([all_y_true, y_true.flatten()])

    th_opt = Thresholdable(all_y_true, all_y_pred)
    thresholds, dices = thresh_exhaustive_eval(th_opt, eps=args.th_step)
    best_idx = np.argmax(dices)
    return thresholds[best_idx], dices[best_idx]


def progress(cls, start, end, _i, n):
    if cls is not None:
        cls.notify_progress((start + (_i / n) * (end - start)))


def main(argv, computation=None):
    """

    IMAGES VALID:
    * 005-TS_13C08351_2-2014-02-12 12.22.44.ndpi | id : 77150767
    * 024-12C07162_2A-2012-08-14-17.21.05.jp2 | id : 77150761
    * 019-CP_12C04234_2-2012-08-10-12.49.26.jp2 | id : 77150809

    IMAGES TEST:
    * 004-PF_08C11886_1-2012-08-09-19.05.53.jp2 | id : 77150623
    * 011-TS_13C10153_3-2014-02-13 15.22.21.ndpi | id : 77150611
    * 018-PF_07C18435_1-2012-08-17-00.55.09.jp2 | id : 77150755

    """
    with Cytomine.connect_from_cli(argv):
        parser = ArgumentParser()
        parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int)
        parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
        parser.add_argument("-e", "--epochs", dest="epochs", default=1, type=int)
        parser.add_argument("-d", "--device", dest="device", default="cpu")
        parser.add_argument("-o", "--tile_overlap", dest="tile_overlap", default=0, type=int)
        parser.add_argument("-t", "--tile_size", dest="tile_size", default=256, type=int)
        parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
        parser.add_argument("-l", "--loss", dest="loss", default="bce", help="['dice','bce','both']")
        parser.add_argument("-r", "--rseed", dest="rseed", default=42, type=int)
        parser.add_argument("-i", "--iter_per_epoch", dest="iter_per_epoch", default=0, type=int)
        parser.add_argument("--dataset", dest="dataset", default="thyroid", help="in ['thyroid', 'monuseg', 'segpc']")
        parser.add_argument("--monu_rr", dest="monuseg_remove_ratio", default=0.0, type=float)
        parser.add_argument("--monu_ms", dest="monuseg_missing_seed", default=42, type=int)
        parser.add_argument("--monu_nc", dest="monuseg_n_complete", default=1, type=int)
        parser.add_argument("--segpc_rr", dest="segpc_remove_ratio", default=0.0, type=float)
        parser.add_argument("--segpc_ms", dest="segpc_missing_seed", default=42, type=int)
        parser.add_argument("--segpc_nc", dest="segpc_n_complete", default=298, type=int)
        parser.add_argument("--lr", dest="lr", default=0.01, type=float)
        parser.add_argument("--init_fmaps", dest="init_fmaps", default=16, type=int)
        parser.add_argument("--save_cues", dest="save_cues", action="store_true")
        parser.add_argument("--th_step", dest="th_step", default=0.01, type=float)
        parser.add_argument("--sparse_start_after", dest="sparse_start_after", default=0, type=int)
        parser.add_argument("--aug_hed_bias_range", dest="aug_hed_bias_range", type=float, default=0.025)
        parser.add_argument("--aug_hed_coef_range", dest="aug_hed_coef_range", type=float, default=0.025)
        parser.add_argument("--aug_blur_sigma_extent", dest="aug_blur_sigma_extent", type=float, default=0.1)
        parser.add_argument("--aug_noise_var_extent", dest="aug_noise_var_extent", type=float, default=0.1)
        parser.add_argument("--lr_sched_factor", dest="lr_sched_factor", type=float, default=0.5)
        parser.add_argument("--lr_sched_patience", dest="lr_sched_patience", type=int, default=3)
        parser.add_argument("--lr_sched_cooldown", dest="lr_sched_cooldown", type=int, default=3)
        parser.add_argument("-wm", "--weights_mode", dest="weights_mode", default="constant")
        parser.add_argument("-wc", "--weights_constant", dest="weights_constant", type=float, default=1.0)
        parser.add_argument("-wf", "--weights_consistency_fn", dest="weights_consistency_fn", default="absolute")
        parser.add_argument("-wn", "--weights_neighbourhood", dest="weights_neighbourhood", type=int, default=1)
        parser.add_argument("-wmin", "--weights_minimum", dest="weights_minimum", type=float, default=0.0)
        parser.add_argument("--sparse_data_rate", dest="sparse_data_rate", type=float, default=1.0, help="<=1.0 = proportion; >1 = number of samples")
        parser.add_argument("--sparse_data_max", dest="sparse_data_max", type=float, default=1.0, help="-1 = same as non sparse; <=1.0 = proportion; >1 = number of samples")
        parser.add_argument("--data_path", "--dpath", dest="data_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("--no_distillation", dest="no_distillation", action="store_true")
        parser.add_argument("--no_groundtruth", dest="no_groundtruth", action="store_true")
        parser.add_argument("-w", "--working_path", "--wpath", dest="working_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-s", "--save_path", dest="save_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-dtm", "--distil_target_mode", dest="distil_target_mode", help="in {'soft', 'hard_dice'}", default="soft")
        parser.add_argument("--n_calibration", dest="n_calibration", type=int, default=1)
        parser.set_defaults(save_cues=False, no_distillation=False, no_groundtruth=False)
        args, _ = parser.parse_known_args(argv)
        print(args)

        if args.no_groundtruth and args.sparse_start_after == -1:
            raise ValueError("no ground truth experiment should start adding sparse data after first epoch")

        if args.no_groundtruth and args.no_distillation:
            raise ValueError("cannot exclude ground truth and distillation at the same time")

        get_random_init_fn(args.rseed)()

        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.data_path, exist_ok=True)
        os.makedirs(args.working_path, exist_ok=True)

        # network
        device = torch.device(args.device)
        unet = Unet(args.init_fmaps, n_classes=1)
        unet.train()
        unet.to(device)

        weights_on = args.loss == "bce" and (args.weights_mode != "constant" or args.weights_constant < 0.99)
        weight_computer = WeightComputer(mode=args.weights_mode, constant_weight=args.weights_constant,
                                         consistency_fn=args.weights_consistency_fn,
                                         consistency_neigh=args.weights_neighbourhood,
                                         min_weight=args.weights_minimum,
                                         logits=True, device=device)

        optimizer = Adam(unet.parameters(), lr=args.lr)
        # stops after five decreases
        mk_sched = partial(ReduceLROnPlateau, mode='min', factor=args.lr_sched_factor, patience=args.lr_sched_patience,
                           threshold=0.005, threshold_mode="abs", cooldown=args.lr_sched_cooldown,
                           min_lr=args.lr * (args.lr_sched_factor ** 5), verbose=True)
        scheduler = mk_sched(optimizer)

        loss_fn = {
            "dice": DiceWithLogitsLoss(reduction="mean"),
            "both": MergedLoss(BCEWithLogitsLoss(reduction="mean"), DiceWithLogitsLoss(reduction="mean"))
        }.get(args.loss, BCEWithWeights(reduction="mean"))

        struct, visual = get_aug_transforms(
            aug_hed_bias_range=args.aug_hed_bias_range,
            aug_hed_coef_range=args.aug_hed_coef_range,
            aug_blur_sigma_extent=args.aug_blur_sigma_extent,
            aug_noise_var_extent=args.aug_noise_var_extent,
            seed=args.rseed)

        trans_dict = {
            "image_trans": visual,
            "both_trans": struct,
            "mask_trans": transforms.ToTensor()
        }

        if args.dataset == "thyroid":
            dataset = ThyroidDatasetGenerator(args.data_path, args.tile_size, args.zoom_level,
                                              n_calibrate=args.n_calibration)
        elif args.dataset == "monuseg":
            dataset = MonusegDatasetGenerator(args.data_path, args.tile_size,
                                              missing_seed=args.monuseg_missing_seed,
                                              remove_ratio=args.monuseg_remove_ratio,
                                              n_complete=args.monuseg_n_complete,
                                              n_calibrate=args.n_calibration)
        elif args.dataset == "segpc":
            dataset = SegpcDatasetGenerator(args.data_path, args.tile_size,
                                            missing_seed=args.segpc_missing_seed,
                                            remove_ratio=args.segpc_remove_ratio,
                                            n_complete=args.segpc_n_complete,
                                            n_calibrate=args.n_calibration)
        else:
            raise ValueError("Unknown dataset '{}'".format(args.dataset))

        incomplete_list, complete_list, val_set_list, calibration_list = dataset.sets()

        # gradual more data
        add_data_state = GraduallyAddMoreDataState(
            incomplete_list, complete_list,
            data_rate=args.sparse_data_rate,
            data_max=args.sparse_data_max)

        complete = dataset.iterable_to_dataset(complete_list, **trans_dict)
        if args.sparse_start_after >= 0:
            incomplete = CropTrainDataset([], **trans_dict)
        else:
            incomplete = dataset.iterable_to_dataset(add_data_state.get_next(), **trans_dict)
        loader_args = {
            "shuffle": True,
            "batch_size": args.batch_size,
            "num_workers": args.n_jobs,
            "worker_init_fn": worker_init,
            "pin_memory": True
        }
        if args.iter_per_epoch > 0:
            loader_args["shuffle"] = False
            sampler_fn = lambda ds: RandomSampler(ds, replacement=True, num_samples=args.iter_per_epoch * args.batch_size)
        else:
            sampler_fn = lambda ds: None

        print("Dataset")
        print("Size: ")
        print("- complete   : {}".format(len(complete_list)))
        print("- sparse     : {}".format(len(incomplete_list)))
        print("- both (bef) : {}".format(len(complete_list) + len(incomplete_list)))
        print("- start (improved) sparse after epoch {}".format(args.sparse_start_after))

        results = {
            "train_losses": [],
            "val_losses": [],
            "val_dice": [],
            "val_soft_dice": [],
            "val_metrics": [],
            "save_path": [],
            "thresholds": [],
            "threshold": []
        }

        for e in range(args.epochs):
            progress_epoch_start = e / args.epochs
            progress_epoch_end = progress_epoch_start + (1 / args.epochs)
            progress(computation, 0, 100, e, args.epochs)
            print("########################")
            print("        Epoch {}".format(e))
            print("########################")

            concat_dataset = ConcatDataset([complete, incomplete])
            print("Training dataset size: {}".format(len(concat_dataset)))
            loader = DataLoader(concat_dataset, sampler=sampler_fn(concat_dataset), **loader_args)

            epoch_losses = list()
            unet.train()
            # crop_loc, crop, gt_mask, cue_mask, mask, has_cue
            for i, (x, y_gt, y_cues, y, has_cues) in enumerate(loader):
                x, y, y_gt, y_cues, has_cues = (t.to(device) for t in [x, y, y_gt, y_cues, has_cues])
                y_pred = unet.forward(x)
                if e != 0 and weights_on and torch.any(has_cues):
                    with torch.no_grad():
                        weights = weight_computer(y_cues.detach(), y_gt.detach(), has_cues)
                    loss = loss_fn(y_pred, y, weights=weights)
                else:
                    loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses = [loss.detach().cpu().item()] + epoch_losses[:5]
                print("{} - {:1.5f}".format(i, np.mean(epoch_losses)))
                results["train_losses"].append(epoch_losses[0])
                # 1/8 of the progress
                progress(computation,
                         progress_epoch_start, progress_epoch_start + (progress_epoch_end - progress_epoch_start) / 8,
                         i + 1, args.iter_per_epoch if args.iter_per_epoch > 0 else len(concat_dataset) / args.batch_size)

            unet.eval()
            # validation
            val_losses = np.zeros(len(val_set_list), dtype=np.float)
            val_roc_auc = np.zeros(len(val_set_list), dtype=np.float)
            val_dice = np.zeros(len(val_set_list), dtype=np.float)

            print("------------------------------")
            print("Eval at epoch {}:".format(e))

            all_y_pred, all_y_true = np.array([]), np.array([])
            no_fg_counter = 0
            for i, roi in enumerate(val_set_list):
                with torch.no_grad():
                    y_pred, y_true = predict_roi(
                        roi, dataset.val_roi_foreground(roi), unet, device,
                        in_trans=get_norm_transform(),
                        batch_size=args.batch_size,
                        tile_size=args.tile_size,
                        overlap=args.tile_overlap,
                        n_jobs=args.n_jobs,
                        zoom_level=args.zoom_level
                    )

                all_y_pred = np.hstack([all_y_pred, y_pred.flatten()])
                all_y_true = np.hstack([all_y_true, y_true.flatten()])

                val_losses[i] = metrics.log_loss(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
                if np.count_nonzero(y_true) == 0:
                    no_fg_counter += 1
                    val_roc_auc[i] = 0
                else:
                    val_roc_auc[i] = metrics.roc_auc_score(y_true.flatten(), y_pred.flatten(), labels=[0, 1])
                val_dice[i] = soft_dice_coefficient(y_true, y_pred)

                progress(computation,
                         progress_epoch_start + (progress_epoch_end - progress_epoch_start) / 8,
                         progress_epoch_start + 3 * (progress_epoch_end - progress_epoch_start) / 8,
                         i + 1, len(val_set_list))

            val_loss = np.mean(val_losses)
            roc_auc = np.mean(val_roc_auc) * len(val_set_list) / (len(val_set_list) - no_fg_counter)

            th_opt = Thresholdable(all_y_true, all_y_pred)
            thresholds, dices = thresh_exhaustive_eval(th_opt, eps=args.th_step)
            results["thresholds"].append((thresholds, dices))
            best_idx = np.argmax(dices)
            threshold, dice = thresholds[best_idx], dices[best_idx]
            cm = metrics.confusion_matrix(all_y_true.astype(np.uint8), (all_y_pred.flatten() > threshold).astype(np.uint8))
            del all_y_pred, all_y_true
            cnt = np.sum(cm)
            soft_dice = np.mean(val_dice)
            print("> val_loss : {:1.5f}".format(val_loss))
            print("> roc_auc  : {:1.5f}".format(roc_auc))
            print("> dice     : {:1.5f}".format(dice))
            print("> soft dice: {:1.5f}".format(soft_dice))
            print("CM at {:0.4f} threshold".format(threshold))
            print("> {:3.2f}%  {:3.2f}%".format(100 * cm[0, 0] / cnt, 100 * cm[0, 1] / cnt))
            print("> {:3.2f}%  {:3.2f}%".format(100 * cm[1, 0] / cnt, 100 * cm[1, 1] / cnt))
            if args.sparse_start_after <= e:
                if args.no_distillation:
                    incomplete = dataset.iterable_to_dataset(add_data_state.get_next(), **trans_dict)
                else:
                    print("------------------------------")
                    print("Improve sparse dataset (after epoch {})".format(args.sparse_start_after))
                    new_crops = predict_annotation_crops_with_cues(
                        unet, add_data_state.get_next(), device, in_trans=get_norm_transform(),
                        overlap=args.tile_overlap, batch_size=args.batch_size, n_jobs=args.n_jobs,
                        progress_fn=partial(progress, computation,
                                            progress_epoch_start + 3 * (progress_epoch_end - progress_epoch_start) / 8,
                                            progress_epoch_start + 7 * (progress_epoch_end - progress_epoch_start) / 8))
                    if args.distil_target_mode == 'hard_dice':
                        if len(calibration_list) == 0:
                            raise ValueError("hard_dice requires setting a positive number of calibration images")
                        calibrated_thresh, calibrated_dice = determine_optimal_threshold(calibration_list, dataset, unet, args, device)
                        print("> calibrated (thresh, dice): {}, {}".format(calibrated_thresh, calibrated_dice))
                        new_crops = [CropWithThresholdedCue(crop, calibrated_thresh * 255) for crop in new_crops]
                    if args.save_cues:
                        cue_save_path = os.path.join(args.data_path, "cues", os.environ.get("SLURM_JOB_ID"), str(e))
                        print("save cues for epoch {} at '{}'".format(e, cue_save_path))
                        save_cues(cue_save_path, new_crops)
                    if args.no_groundtruth:
                        for awcue in new_crops:
                            awcue.cue_only = True
                    incomplete = dataset.iterable_to_dataset(new_crops, **trans_dict)
                    print("------------------------------")

            # reset scheduler when adding new samples
            if args.sparse_start_after == e:
                scheduler = mk_sched(optimizer)
            else:
                scheduler.step(dice)

            filename = "{}_e_{}_val_{:0.4f}_roc_{:0.4f}_z{}_s{}.pth".format(datetime.now().timestamp(), e, val_loss, roc_auc, args.zoom_level, args.tile_size)
            torch.save(unet.state_dict(), os.path.join(args.save_path, filename))

            results["val_losses"].append(val_loss)
            results["val_dice"].append(dice)
            results["val_soft_dice"].append(soft_dice)
            results["val_metrics"].append(roc_auc)
            results["save_path"].append(filename)
            results["threshold"].append(threshold)

        return results


class TrainComputation(Computation):
    def __init__(self, exp_name, comp_name, host=None, private_key=None, public_key=None, n_jobs=1, device="cuda:0",
                 save_path=None, data_path=None, th_step=0.01, context="n/a", storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._n_jobs = n_jobs
        self._device = device
        self._save_path = save_path
        self._cytomine_private_key = private_key
        self._cytomine_public_key = public_key
        self._cytomine_host = host
        self._data_path = data_path
        self._th_step = th_step

    def run(self, results, batch_size=4, epochs=4, overlap=0, tile_size=512, lr=.001, init_fmaps=16, zoom_level=0,
            sparse_start_after=0, aug_hed_bias_range=0.025, aug_hed_coef_range=0.025, aug_blur_sigma_extent=0.1,
            aug_noise_var_extent=0.1, save_cues=False, loss="bce", lr_sched_factor=0.5, lr_sched_patience=3,
            lr_sched_cooldown=3, sparse_data_max=1.0, sparse_data_rate=1.0, no_distillation=False,
            no_groundtruth=False, weights_mode="constant", weights_constant=1.0, weights_consistency_fn="absolute",
            weights_neighbourhood=1, rseed=42, weights_minimum=0.0, dataset="thyroid", monu_rr=0.0, monu_ms=42,
            monu_nc=1, iter_per_epoch=0, distil_target_mode="soft", n_calibration=0, segpc_rr=0.0, segpc_ms=42,
            segpc_nc=298):
        # import os
        # os.environ['MKL_THREADING_LAYER'] = 'GNU'
        argv = ["--host", str(self._cytomine_host),
                "--public_key", str(self._cytomine_public_key),
                "--private_key", str(self._cytomine_private_key),
                "--batch_size", str(batch_size),
                "--n_jobs", str(self._n_jobs),
                "--epochs", str(epochs),
                "--rseed", str(rseed),
                "--device", str(self._device),
                "--tile_overlap", str(overlap),
                "--tile_size", str(tile_size),
                "--lr", str(lr),
                "--init_fmaps", str(init_fmaps),
                "--data_path", str(self._data_path),
                "--working_path", os.path.join(os.environ["SCRATCH"], os.environ["SLURM_JOB_ID"]),
                "--save_path", str(self._save_path),
                "--loss", str(loss),
                "--zoom_level", str(zoom_level),
                "--sparse_start_after", str(sparse_start_after),
                "--aug_hed_bias_range", str(aug_hed_bias_range),
                "--aug_hed_coef_range", str(aug_hed_coef_range),
                "--aug_blur_sigma_extent", str(aug_blur_sigma_extent),
                "--aug_noise_var_extent", str(aug_noise_var_extent),
                "--lr_sched_factor", str(lr_sched_factor),
                "--lr_sched_patience", str(lr_sched_patience),
                "--lr_sched_cooldown", str(lr_sched_cooldown),
                "--sparse_data_rate", str(sparse_data_rate),
                "--sparse_data_max", str(sparse_data_max),
                "--weights_mode", str(weights_mode),
                "--weights_constant", str(weights_constant),
                "--weights_consistency_fn", str(weights_consistency_fn),
                "--weights_neighbourhood", str(weights_neighbourhood),
                "--weights_minimum", str(weights_minimum),
                "--dataset", str(dataset),
                "--monu_rr", str(monu_rr),
                "--monu_ms", str(monu_ms),
                "--monu_nc", str(monu_nc),
                "--iter_per_epoch", str(iter_per_epoch),
                "--th_step", str(self._th_step),
                "--distil_target_mode", str(distil_target_mode),
                "--n_calibration", str(n_calibration),
                "--segpc_rr", str(segpc_rr),
                "--segpc_ms", str(segpc_ms),
                "--segpc_nc", str(segpc_nc)]
        if save_cues:
            argv.append("--save_cues")
        if no_distillation:
            argv.append("--no_distillation")
        if no_groundtruth:
            argv.append("--no_groundtruth")
        for k, v in main(argv, computation=self).items():
            results[k] = v


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])