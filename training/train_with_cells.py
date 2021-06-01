import os
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import joblib
import torch
import numpy as np

from clustertools import Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from joblib import delayed
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import box
from skimage import io
from sklearn import metrics
from sldc import batch_split
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms

from augment import get_aug_transforms, get_norm_transform
from dataset import RemoteAnnotationCropTrainDataset, predict_roi, AnnotationCrop, AnnotationCropWithCue, \
    predict_annotation_crops_with_cues
from thyroid import get_thyroid_annotations, get_pattern_train, get_val_set, VAL_TEST_IDS, VAL_IDS, get_cell_train
from unet import Unet, DiceWithLogitsLoss, MergedLoss


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


def generic_match_search(key_item, elements, item_fn, match_fn):
    return [elem for elem in elements if match_fn(key_item, item_fn(elem))]


def get_polygons_intersecting_crop_roi(crop, crops):
    (x, y), width, height = crop.image_box
    bbox = box(x, y, x + width, y + height)
    return generic_match_search(bbox, crops,
                                item_fn=lambda c: c.polygon,
                                match_fn=lambda a, b: a.intersects(b))


def worker_init(tid):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def _parallel_download_wsi(identifiers, path, argv):
    with Cytomine.connect_from_cli(argv):
        instances = list()
        for _id in identifiers:
            instance = ImageInstance().fetch(_id)
            filepath = os.path.join(path, instance.originalFilename)
            print("Download", filepath)
            instance.download(dest_pattern=filepath, override=False)
            instance.download_path = filepath
            instances.append(instance)
        return instances


def download_wsi(path, identifiers, argv, n_jobs=1):
    batches = batch_split(n_jobs, identifiers)
    results = joblib.Parallel(n_jobs)(delayed(_parallel_download_wsi)(batch, path, argv) for batch in batches)
    return [i for b in results for i in b]


def polyref_proc2cyto(p, height_at_zoom_level, zoom_level=0):
    p = affine_transform(p, [1, 0, 0, -1, 0, height_at_zoom_level])
    return affine_transform(p, [2 ** zoom_level, 0, 0, 2 ** zoom_level, 0, 0])


def save_cues(path, crops_with_cues):
    os.makedirs(path, exist_ok=True)
    for crop in crops_with_cues:
        io.imsave(os.path.join(path, crop.annotation.originalFilename), crop.cue)


def group_annot_per_image(annots):
    dd = defaultdict(list)
    for a in annots:
        dd[a.image].append(a)
    return dd

class GraduallyAddMoreDataState(object):
    def __init__(self, sparse, non_sparse, data_rate=1.0, data_max=1.0):
        self._data_rate = data_rate
        self._data_max = data_max
        self._sparse = sparse
        self._non_sparse = non_sparse
        self._current_amount = 0

    @property
    def abs_data_max(self):
        if self._data_max < 0:
            return min(len(self._non_sparse), len(self._sparse))
        elif 0 <= self._data_max <= 1:
            return int(self._data_max * len(self._sparse))
        else:
            return min(int(self._data_max), len(self._sparse))

    @property
    def abs_date_to_add(self):
        if 0 <= self._data_rate <= 1:
            return int(self._data_rate * len(self._sparse))
        elif self._data_rate > 1:
            return int(self._data_max)

    def get_next(self):
        self._current_amount += self.abs_date_to_add
        self._current_amount = min(self._current_amount, self.abs_data_max)
        return self._sparse[:self._current_amount]



def main(argv):
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
        parser.add_argument("--lr", dest="lr", default=0.01, type=float)
        parser.add_argument("--init_fmaps", dest="init_fmaps", default=16, type=int)
        parser.add_argument("--save_cues", dest="save_cues", action="store_true")
        parser.add_argument("--sparse_start_after", dest="sparse_start_after", default=0, type=int)
        parser.add_argument("--aug_hed_bias_range", dest="aug_hed_bias_range", type=float, default=0.025)
        parser.add_argument("--aug_hed_coef_range", dest="aug_hed_coef_range", type=float, default=0.025)
        parser.add_argument("--aug_blur_sigma_extent", dest="aug_blur_sigma_extent", type=float, default=0.1)
        parser.add_argument("--aug_noise_var_extent", dest="aug_noise_var_extent", type=float, default=0.1)
        parser.add_argument("--lr_sched_factor", dest="lr_sched_factor", type=float, default=0.5)
        parser.add_argument("--lr_sched_patience", dest="lr_sched_patience", type=int, default=3)
        parser.add_argument("--lr_sched_cooldown", dest="lr_sched_cooldown", type=int, default=3)
        parser.add_argument("--sparse_data_rate", dest="sparse_data_rate", type=float, default=1.0, help="<=1.0 = proportion; >1 = number of samples")
        parser.add_argument("--sparse_data_max", dest="sparse_data_max", type=float, default=1.0, help="-1 = same as non sparse; <=1.0 = proportion; >1 = number of samples")
        parser.add_argument("--data_path", "--dpath", dest="data_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-w", "--working_path", "--wpath", dest="working_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-s", "--save_path", dest="save_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.set_defaults(save_cues=False)
        args, _ = parser.parse_known_args(argv)
        print(args)

        get_random_init_fn(args.rseed)()

        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.data_path, exist_ok=True)
        os.makedirs(args.working_path, exist_ok=True)

        # fetch annotations (filter val/test sets + other annotations)
        all_annotations = get_thyroid_annotations()
        pattern_collec = get_pattern_train(all_annotations)
        cell_collec = get_cell_train(all_annotations)
        train_collec = pattern_collec + cell_collec
        val_rois, val_foreground = get_val_set(all_annotations)
        train_wsi_ids = list({an.image for an in all_annotations}.difference(VAL_TEST_IDS))
        val_wsi_ids = list(VAL_IDS)

        download_path = os.path.join(args.data_path, "crops-{}".format(args.tile_size))
        images = {_id: ImageInstance().fetch(_id) for _id in (train_wsi_ids + val_wsi_ids)}

        print("find crops intersecting ROIs")
        match_params = {
            "item_fn": lambda a: wkt.loads(a.location),
            "match_fn": lambda a, b: a.intersects(b)
        }
        print("base cell crops... ", end="", flush=True)
        annots_per_image = group_annot_per_image(train_collec)
        intersecting = {
            annot.id: generic_match_search(
                key_item=box(*wkt.loads(annot.location).bounds),
                elements=annots_per_image[annot.image],
                **match_params)
            for annot in train_collec
        }
        print("done")
        print("validation rois... ", end="", flush=True)
        val_rois_to_intersect = {
            roi.id: generic_match_search(
                key_item=wkt.loads(roi.location),
                elements=[a for a in val_foreground if a.image == roi.image],
                **match_params)
            for roi in val_rois
        }
        print("done")

        pattern_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, args.tile_size,
            zoom_level=args.zoom_level, intersecting=intersecting[annot.id]) for annot in pattern_collec]
        base_cell_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, args.tile_size,
            zoom_level=args.zoom_level, intersecting=intersecting[annot.id]) for annot in cell_collec]
        val_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, args.tile_size,
            zoom_level=args.zoom_level) for annot in val_rois]

        for crop in pattern_crops + base_cell_crops + val_crops:
            crop.download()

        # network
        device = torch.device(args.device)
        unet = Unet(args.init_fmaps, n_classes=1)
        unet.train()
        unet.to(device)

        optimizer = Adam(unet.parameters(), lr=args.lr)
        # stops after five decreases
        mk_sched = partial(ReduceLROnPlateau, mode='min', factor=args.lr_sched_factor, patience=args.lr_sched_patience,
                           threshold=0.005, threshold_mode="abs", cooldown=args.lr_sched_cooldown,
                           min_lr=args.lr * (args.lr_sched_factor ** 5), verbose=True)
        scheduler = mk_sched(optimizer)

        loss_fn = {
            "dice": DiceWithLogitsLoss(reduction="mean"),
            "both": MergedLoss(BCEWithLogitsLoss(reduction="mean"), DiceWithLogitsLoss(reduction="mean"))
        }.get(args.loss, BCEWithLogitsLoss(reduction="mean"))

        results = {
            "train_losses": [],
            "val_losses": [],
            "val_dice": [],
            "val_metrics": [],
            "save_path": []
        }

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

        # gradual more data
        add_data_state = GraduallyAddMoreDataState(
            base_cell_crops, pattern_crops,
            data_rate=args.sparse_data_rate,
            data_max=args.sparse_data_max)

        full_dataset = RemoteAnnotationCropTrainDataset(pattern_crops, **trans_dict)
        if args.sparse_start_after >= 0:
            sparse_dataset = RemoteAnnotationCropTrainDataset([], **trans_dict)
        else:
            sparse_dataset = RemoteAnnotationCropTrainDataset(add_data_state.get_next(), **trans_dict)

        print("Dataset")
        print("Size: ")
        print("- complete   : {}".format(len(pattern_crops)))
        print("- sparse     : {}".format(len(base_cell_crops)))
        print("- both (bef) : {}".format(len(full_dataset)))
        print("- start (improved) sparse after epoch {}".format(args.sparse_start_after))

        for e in range(args.epochs):
            print("########################")
            print("        Epoch {}".format(e))
            print("########################")

            concat_dataset = ConcatDataset([sparse_dataset, full_dataset])
            loader = DataLoader(concat_dataset, shuffle=True, batch_size=args.batch_size,
                                num_workers=args.n_jobs, worker_init_fn=worker_init)

            epoch_losses = list()
            unet.train()
            for i, (x, y) in enumerate(loader):
                x, y = (t.to(device) for t in [x, y])
                y_pred = unet.forward(x)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses = [loss.detach().cpu().item()] + epoch_losses[:5]
                print("{} - {:1.5f}".format(i, np.mean(epoch_losses)))
                results["train_losses"].append(epoch_losses[0])

            unet.eval()
            # validation
            val_losses = np.zeros(len(val_rois), dtype=np.float)
            val_roc_auc = np.zeros(len(val_rois), dtype=np.float)
            val_dice = np.zeros(len(val_rois), dtype=np.float)
            val_cm = np.zeros([len(val_rois), 2, 2], dtype=np.int)

            print("------------------------------")
            print("Eval at epoch {}:".format(e))

            for i, roi in enumerate(val_crops):
                with torch.no_grad():
                    y_pred, y_true = predict_roi(
                        roi, val_rois_to_intersect[roi.annotation.id], unet, device,
                        in_trans=get_norm_transform(),
                        batch_size=args.batch_size,
                        tile_size=args.tile_size,
                        overlap=args.tile_overlap,
                        n_jobs=args.n_jobs,
                        zoom_level=args.zoom_level
                    )

                val_losses[i] = metrics.log_loss(y_true.flatten(), y_pred.flatten())
                val_roc_auc[i] = metrics.roc_auc_score(y_true.flatten(), y_pred.flatten())
                val_dice[i] = soft_dice_coefficient(y_true, y_pred)
                val_cm[i] = metrics.confusion_matrix(y_true.flatten().astype(np.uint8), (y_pred.flatten() > 0.5).astype(np.uint8))

            val_loss = np.mean(val_losses)
            roc_auc = np.mean(val_roc_auc)
            dice = np.mean(val_dice)
            print("> val_loss: {:1.5f}".format(val_loss))
            print("> roc_auc : {:1.5f}".format(roc_auc))
            print("> dice    : {:1.5f}".format(dice))
            cm = np.sum(val_cm, axis=0)
            cnt = np.sum(val_cm)
            print("CM at 0.5 threshold")
            print("> {:3.2f}%  {:3.2f}%".format(100 * cm[0, 0] / cnt, 100 * cm[0, 1] / cnt))
            print("> {:3.2f}%  {:3.2f}%".format(100 * cm[1, 0] / cnt, 100 * cm[1, 1] / cnt))
            if args.sparse_start_after <= e:
                print("------------------------------")
                print("Improve sparse dataset (after epoch {})".format(args.sparse_start_after))
                new_crops = predict_annotation_crops_with_cues(
                    unet, add_data_state.get_next(), device, in_trans=get_norm_transform(),
                    overlap=args.tile_overlap, batch_size=args.batch_size, n_jobs=args.n_jobs)
                if args.save_cues:
                    cue_save_path = os.path.join(args.data_path, "cues", os.environ.get("SLURM_JOB_ID"), str(e))
                    print("save cues for epoch {} at '{}'".format(e, cue_save_path))
                    save_cues(cue_save_path, new_crops)
                sparse_dataset = RemoteAnnotationCropTrainDataset(new_crops, **trans_dict)
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
            results["val_metrics"].append(roc_auc)
            results["save_path"].append(filename)

        return results


class TrainComputation(Computation):
    def __init__(self, exp_name, comp_name, host=None, private_key=None, public_key=None, n_jobs=1, device="cuda:0",
                 save_path=None, data_path=None, context="n/a", storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._n_jobs = n_jobs
        self._device = device
        self._save_path = save_path
        self._cytomine_private_key = private_key
        self._cytomine_public_key = public_key
        self._cytomine_host = host
        self._data_path = data_path

    def run(self, results, batch_size=4, epochs=4, overlap=0, tile_size=512, lr=.001, init_fmaps=16, zoom_level=0,
            sparse_start_after=0, aug_hed_bias_range=0.025, aug_hed_coef_range=0.025, aug_blur_sigma_extent=0.1,
            aug_noise_var_extent=0.1, save_cues=False, loss="bce", lr_sched_factor=0.5, lr_sched_patience=3,
            lr_sched_cooldown=3, sparse_data_max=1.0, sparse_data_rate=1.0):
        # import os
        # os.environ['MKL_THREADING_LAYER'] = 'GNU'
        argv = ["--host", str(self._cytomine_host),
                "--public_key", str(self._cytomine_public_key),
                "--private_key", str(self._cytomine_private_key),
                "--batch_size", str(batch_size),
                "--n_jobs", str(self._n_jobs),
                "--epochs", str(epochs),
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
                "--sparse_data_max", str(sparse_data_max)]
        if save_cues:
            argv.append("--save_cues")
        for k, v in main(argv).items():
            results[k] = v


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])