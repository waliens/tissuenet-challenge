import os
import re
from argparse import ArgumentParser

import numpy as np
import torch
from clustertools import build_datacube
from cytomine import Cytomine
from cytomine.models import Annotation, ImageInstance, AnnotationCollection
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import box
from skimage import io
from torchvision.transforms import transforms

from dataset import AnnotationCrop, predict_annotation_crops_with_cues
from unet import Unet
from weight_generator import WeightComputer

CDEGAND_ID = 55502856
MTESTOURI_ID = 142954314

PATTERN_TERMS = {35777351, 35777321, 35777459}
CELL_TERMS = {35777375, 35777387, 35777345, 35777441, 35777393, 35777447, 35777339}
VAL_FOREGROUND_TERMS = {154005477}


def get_norm_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

"""77242961,77255703,77201101,77257487,77221903"""
def save_cues(path, crops_with_cues, prefix):
    os.makedirs(path, exist_ok=True)
    for crop in crops_with_cues:
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + "_cue.png"), crop.cue)
        c, m = crop.crop_and_mask()
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + ".png"), np.asarray(c))
        m = np.asarray(m)
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + "_mask_cue.png"), m)
        _, bm = crop._crop.crop_and_mask()
        bm = np.asarray(bm)
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + "_mask.png"), bm)

        colored = np.zeros(m.shape + (3,), dtype=np.uint8)
        colored[:, :, 0] = np.maximum(crop.cue, bm).astype(np.uint8)
        colored[:, :, 1] = bm.astype(np.uint8)
        colored[:, :, 2] = bm.astype(np.uint8)
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + "_rgb.png"), colored)

        alpha = np.zeros(m.shape + (4,), dtype=np.uint8)
        alpha[:, :, :3] = np.asarray(c).astype(np.uint8)
        alpha[:, :, -1] = m.astype(np.uint8)
        io.imsave(os.path.join(path, prefix + "_" + str(crop._crop.annotation.id) + "_alpha.png"), alpha)


def save_weights(path, crops_with_cues, prefix):
    all_params = [
        ("balance_gt", {"mode": "balance_gt", "constant_weight": 1.0, "consistency_fn": None, "consistency_neigh": 1}),
        ("pred_entropy", {"mode": "pred_entropy", "constant_weight": 1.0, "consistency_fn": None, "consistency_neigh": 1}),
        ("pred_consistency_8_abs", {"mode": "pred_consistency", "constant_weight": 1.0, "consistency_fn": "absolute", "consistency_neigh": 1}),
        ("pred_merged_8_abs", {"mode": "pred_merged", "constant_weight": 1.0, "consistency_fn": "absolute", "consistency_neigh": 1}),
        ("pred_consistency_16_abs", {"mode": "pred_consistency", "constant_weight": 1.0, "consistency_fn": "absolute", "consistency_neigh": 2}),
        ("pred_merged_16_abs", {"mode": "pred_merged", "constant_weight": 1.0, "consistency_fn": "absolute", "consistency_neigh": 2}),
        ("pred_consistency_8_quad", {"mode": "pred_consistency", "constant_weight": 1.0, "consistency_fn": "quadratic", "consistency_neigh": 1}),
        ("pred_merged_8_quad", {"mode": "pred_merged", "constant_weight": 1.0, "consistency_fn": "quadratic", "consistency_neigh": 1}),
        ("pred_consistency_16_quad", {"mode": "pred_consistency", "constant_weight": 1.0, "consistency_fn": "quadratic", "consistency_neigh": 2}),
        ("pred_merged_16_quad", {"mode": "pred_merged", "constant_weight": 1.0, "consistency_fn": "quadratic", "consistency_neigh": 2})
    ]
    to_tensor = transforms.ToTensor()

    for name, params in all_params:
        computer = WeightComputer(**params)
        for crop_with_cues in crops_with_cues:
            _, mask = crop_with_cues.crop.crop_and_mask()
            cue = crop_with_cues.cue
            y_gt = to_tensor(mask)
            y = to_tensor(cue / 255)
            w = computer(y, y_gt)
            io.imsave(os.path.join(path, prefix + "_" + str(crop_with_cues.crop.annotation.id) + "_w_" + name + ".png"), (w.detach().cpu().numpy()[0] * 255).astype(np.uint8))


def get_folder(**kwargs):
    args = [
        "tile_size",
        "zoom_level",
        "loss",
        "sparse_start_after",
        "sparse_data_rate",
        "sparse_data_max"
    ]
    return "_".join([str(kwargs[a]) for a in args])


def main(argv):
    with Cytomine.connect_from_cli(argv):
        parser = ArgumentParser()
        parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int)
        parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
        parser.add_argument("-d", "--device", dest="device", default="cpu")
        parser.add_argument("-o", "--tile_overlap", dest="tile_overlap", default=0, type=int)
        parser.add_argument("-t", "--tile_size", dest="tile_size", default=256, type=int)
        parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
        parser.add_argument("-r", "--rseed", dest="rseed", default=42, type=int)
        parser.add_argument("-e", "--epoch", dest="epoch")
        parser.add_argument("-a", "--annotations", dest="annotations")
        parser.add_argument("--save_path", dest="save_path")
        parser.add_argument("--data_path", dest="data_path")
        parser.add_argument("--model_path", dest="model_path")
        parser.add_argument("--sdr", dest="sdr")
        parser.add_argument("--sdm", dest="sdm")
        parser.add_argument("--loss", dest="loss")
        parser.add_argument("--ssa", dest="ssa", type=int)
        parser.set_defaults(save_cues=False)
        args, _ = parser.parse_known_args(argv)
        print(args)

        download_path = os.path.join(args.data_path, "crops-{}".format(args.tile_size))
        save_folder = get_folder(tile_size=args.tile_size, zoom_level=args.zoom_level, loss=args.loss,
                                 sparse_start_after=args.ssa, sparse_data_rate=args.sdr, sparse_data_max=args.sdm)
        save_path = os.path.join(args.save_path, save_folder)
        os.makedirs(save_path, exist_ok=True)

        datacube = build_datacube("thyroid-unet-training-study")
        dc_slice = datacube(
            loss=str(args.loss), no_distillation=False, no_groundtruth=False, # tile_size=str(args.tile_size), zoom_level=str(args.zoom_level),
            sparse_data_max=str(args.sdm), sparse_data_rate=str(args.sdr), sparse_start_after=str(args.ssa))

        models = dc_slice["save_path"]
        pattern = re.compile(".*_e_{}_.*".format(args.epoch))
        filtered = [m for m in models if pattern.match(m) is not None]

        if len(filtered) != 1:
            raise ValueError("no model for given epoch")

        all_crops = list()
        for annot_id in list(map(int, args.annotations.split(","))):
            annotation = Annotation().fetch(annot_id)
            image_instance = ImageInstance().fetch(annotation.image)
            crop_kwargs = {"working_path": download_path, "tile_size": args.tile_size, "zoom_level": args.zoom_level, "n_jobs": args.n_jobs}
            crop_args = [image_instance, annotation]
            crop = AnnotationCrop(*crop_args, **crop_kwargs)
            (x, y), width, height = crop.image_box
            roi = affine_transform(affine_transform(box(x, y, x + width, y + height), [1, 0, 0, -1, 0, image_instance.height]), [1 / (2 ** args.zoom_level), 0, 0, 1 / (2 ** args.zoom_level), 0, 0])
            in_image = AnnotationCollection(
                users=[CDEGAND_ID, MTESTOURI_ID],
                terms=list(VAL_FOREGROUND_TERMS.union(PATTERN_TERMS).union(CELL_TERMS)),
                image=image_instance.id, showWKT=True, showMeta=True).fetch()
            intersecting = [a for a in in_image if wkt.loads(a.location).intersects(roi) and a.id != annot_id]
            final_crop = AnnotationCrop(*crop_args, **crop_kwargs, intersecting=intersecting)

            all_crops.append(final_crop)

        device = torch.device(args.device)
        unet = Unet(int(dc_slice.metadata["init_fmaps"]), n_classes=1)
        unet.load_state_dict(torch.load(os.path.join(args.model_path, filtered[0]), map_location=device))
        unet.to(device)
        unet.eval()

        print("------------------------------")
        print("Stat generate cues)")
        with torch.no_grad():
            new_crops = predict_annotation_crops_with_cues(
                unet, all_crops, device, in_trans=get_norm_transform(),
                overlap=args.tile_overlap, batch_size=args.batch_size, n_jobs=args.n_jobs)
            save_weights(save_path, new_crops, prefix="e_{}".format(args.epoch))
            save_cues(save_path, new_crops, prefix="e_{}".format(args.epoch))


if __name__ == "__main__":
    import sys

    for i in range(50):
        main(sys.argv[1:] + ["--epoch", str(i)])
