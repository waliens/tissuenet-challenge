import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageColor
from clustertools import build_datacube, Experiment, Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine

from augment import get_norm_transform
from dataset import predict_roi
from monuseg import MonusegDatasetGenerator
from thyroid import ThyroidDatasetGenerator
from unet import Unet


def image_with_metadata(img, font_size=40, font_size_small=30, margin=10, v_offset=0, **data):
    if v_offset == 0:
        v_offset = - (font_size * 2 + font_size_small + 4 * margin)
    h, w = img.shape
    final = np.zeros([h - v_offset, w], np.uint8)
    final[:h] = img
    final[h:] = 225
    final[h:h+5, ::2] = 0
    final[h:h+5, 1::2] = 255

    font_path = "/home/rmormont/fonts/LiberationMono-Regular.ttf"
    font = ImageFont.truetype(font_path, font_size)
    small_font = ImageFont.truetype(font_path, font_size_small)
    pil_final = Image.fromarray(final)
    draw_final = ImageDraw.Draw(pil_final)

    base_x, base_y = margin, final.shape[0] + v_offset + margin
    row2_y = base_y + margin + font_size
    row3_y = row2_y + margin + margin // 2 + font_size_small

    draw_final.text((base_x, base_y), "e = {}".format(data["epoch"]), fill=0, font=font)
    draw_final.text((base_x + 200, base_y), "dce = {:0.4f}".format(data["val_dice"]), fill=0, font=font)
    draw_final.text((base_x + 525, base_y), "roc = {:0.4f}".format(data["val_roc"]), fill=0, font=font)
    if data["best"]:
        draw_final.text((base_x + 825, base_y), "(best)", fill=0, font=font)
    draw_final.text((base_x, base_y + margin + font_size), "t = {:0.4f}".format(data["thresh"]), fill=0, font=font)
    draw_final.text((base_x + 300, row2_y), "{}".format(data["file"]), fill=0, font=font)

    draw_final.text((base_x, row3_y), "nc={}".format(data["nc"]), fill=0, font=small_font)
    draw_final.text((base_x + 150, row3_y), "rr={}".format(data["rr"]), fill=0, font=small_font)
    draw_final.text((base_x + 300, row3_y), "d={}".format(data["d"]), fill=0, font=small_font)
    draw_final.text((base_x + 450, row3_y), "ssa={}".format(data["ssa"]), fill=0, font=small_font)
    draw_final.text((base_x + 600, row3_y), "w={}".format(data["w"]), fill=0, font=small_font)

    return np.asarray(pil_final)


def main(argv):
    print(argv)
    with Cytomine.connect_from_cli(argv):
        parser = ArgumentParser()
        parser.add_argument("-b", "--batch_size", dest="batch_size", default=4, type=int)
        parser.add_argument("-o", "--tile_overlap", dest="tile_overlap", default=0, type=int)
        parser.add_argument("-t", "--tile_size", dest="tile_size", default=256, type=int)
        parser.add_argument("-j", "--n_jobs", dest="n_jobs", default=1, type=int)
        parser.add_argument("-d", "--device", dest="device", default="cpu")
        parser.add_argument("-z", "--zoom_level", dest="zoom_level", default=0, type=int)
        parser.add_argument("--dataset", dest="dataset", default="thyroid", help="in ['thyroid', 'monuseg', 'pannuke']")
        parser.add_argument("-i", "--image", dest="image", required=True)
        parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=0)
        parser.add_argument("--model_dir", dest="model_dir", default=".")
        parser.add_argument("--data_path", dest="data_path", default=".")
        parser.add_argument("--save_path", dest="save_path", default=".")
        parser.add_argument("--experiment", dest="experiment")
        parser.add_argument("--exp_rr", dest="exp_rr", default=0.0, type=float)
        parser.add_argument("--exp_ms", dest="exp_ms", default=42, type=int)
        parser.add_argument("--exp_nc", dest="exp_nc", default=1, type=int)
        parser.add_argument("--exp_weights_mode", dest="exp_weights_mode", default="constant")
        parser.add_argument("--exp_sparse_start_after", dest="exp_sparse_start_after", default=0, type=int)
        parser.add_argument("--exp_no_distillation", dest="exp_no_distillation", action="store_true")
        parser.set_defaults(exp_no_distillation=False)
        args, _ = parser.parse_known_args(argv)

        if args.dataset == "thyroid":
            dataset = ThyroidDatasetGenerator(args.data_path, args.tile_size, args.zoom_level)
        elif args.dataset == "monuseg":
            dataset = MonusegDatasetGenerator(args.data_path, args.tile_size, remove_ratio=0, n_complete=30, missing_seed=42)
        else:
            raise ValueError("Unknown dataset '{}'".format(args.dataset))

        # model
        device = torch.device(args.device)

        # extracting experiment
        cube = build_datacube(args.experiment)
        cube = cube(
            monu_ms=str(args.exp_ms), monu_nc=str(args.exp_nc),
            monu_rr=str(args.exp_rr),
            no_distillation=str(args.exp_no_distillation),
            sparse_start_after=str(args.exp_sparse_start_after),
            weights_mode=str(args.exp_weights_mode),
        )

        # model
        model_filename = cube("save_path")[args.epoch]
        model_filepath = os.path.join(args.model_dir, model_filename)
        unet = Unet(int(cube.metadata["init_fmaps"]), n_classes=1)
        unet.load_state_dict(torch.load(model_filepath, map_location=device))
        unet.to(device)
        unet.eval()

        # data
        crop = dataset.crop(args.image)

        y_pred, y_true = predict_roi(
            crop, dataset.val_roi_foreground(crop), unet, device,
            in_trans=get_norm_transform(),
            batch_size=args.batch_size,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            n_jobs=args.n_jobs,
            zoom_level=args.zoom_level
        )

        output_filepath = os.path.join(args.save_path, "{:02d}_{}_{}_{}_{}_{}_{}_".format(
            args.epoch, args.exp_ms,
            args.exp_nc, args.exp_rr,
            args.exp_no_distillation,
            args.exp_sparse_start_after,
            args.exp_weights_mode
        ) + args.image.replace(".tif", "_{}.png"))

        thresh = cube("threshold")[args.epoch]
        metadata = {
            "epoch": args.epoch,
            "val_dice": cube("val_dice")[args.epoch],
            "val_roc": cube("val_metrics")[args.epoch],
            "file": args.image,
            "best": np.argmax(cube("val_dice")) == args.epoch,
            "thresh": cube("threshold")[args.epoch],
            "nc": args.exp_nc,
            "rr": args.exp_rr,
            "d": not args.exp_no_distillation,
            "ssa": args.exp_sparse_start_after,
            "w": args.exp_weights_mode,
        }

        cv2.imwrite(output_filepath.format("prob"), image_with_metadata((y_pred * 255).astype(np.uint8), **metadata))
        cv2.imwrite(output_filepath.format("thre"), image_with_metadata((y_pred > thresh).astype(np.uint8) * 255, **metadata))
        return dict()


class ApplyModelComputation(Computation):
    def __init__(self, exp_name, comp_name, host=None, private_key=None, public_key=None, n_jobs=1, device="cuda:0",
                 save_path=None, data_path=None, model_dir=None, context="n/a", storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._n_jobs = n_jobs
        self._device = device
        self._save_path = save_path
        self._data_path = data_path
        self._model_dir = model_dir
        self._cytomine_private_key = private_key
        self._cytomine_public_key = public_key
        self._cytomine_host = host

    def run(self, results, **params):
        # batch_size
        # tile_overlap
        # tile_size
        # zoom_level
        # dataset
        # image
        # epoch
        # output
        # experiment
        # exp_rr
        # exp_ms
        # exp_nc
        # exp_weights_mode
        # exp_sparse_start_after
        # exp_no_distillation
        argv = [("--{}".format(p), str(v)) for p, v in params.items() if not isinstance(v, bool)]
        argv = [i for t in argv for i in t]  # flatten tupple array

        # add boolean
        argv += ["--{}".format(p) for p, v in params.items() if isinstance(v, bool) and v]

        argv += ["--n_jobs", str(self._n_jobs), "--device", self._device, "--data_path", self._data_path,
                 "--save_path", self._save_path, "--model_dir", self._model_dir,
                 "--private_key", self._cytomine_private_key, "--public_key", self._cytomine_public_key,
                 "--host", self._cytomine_host]
        for k, v in main(argv).items():
            results[k] = v


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])