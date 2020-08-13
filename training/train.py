import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch


from clustertools import Computation
from clustertools.storage import PickleStorage
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from joblib import delayed
from shapely import wkt
from sklearn import metrics
from sldc import batch_split
from sldc_cytomine import CytomineSlide
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import RemoteAnnotationTrainDataset, segmentation_transform, predict_roi, AnnotationCrop
from unet import Unet


def torange0_1(t):
    return t / 255.0


def find_intersecting_annotations(roi, annotations):
    found = list()
    roi_location = wkt.loads(roi.location)
    for annot in annotations:
        location = wkt.loads(annot.location)
        if roi_location.intersects(location):
            found.append(annot)
    return found


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
        parser.add_argument("-o", "--overlap", dest="overlap", default=0, type=int)
        parser.add_argument("-t", "--tile_size", dest="tile_size", default=256, type=int)
        parser.add_argument("--lr", dest="lr", default=0.01, type=float)
        parser.add_argument("--init_fmaps", dest="init_fmaps", default=16, type=int)
        parser.add_argument("--data_path", "--dpath", dest="data_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-w", "--working_path", "--wpath", dest="working_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        parser.add_argument("-s", "--save_path", dest="save_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        args, _ = parser.parse_known_args(argv)

        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.data_path, exist_ok=True)
        os.makedirs(args.working_path, exist_ok=True)

        # fetch annotations (filter val/test sets + other annotations)
        all_annotations = AnnotationCollection(project=77150529, showWKT=True, showMeta=True, showTerm=True).fetch()
        val_ids = {77150767, 77150761, 77150809}
        test_ids = {77150623, 77150611, 77150755}
        val_test_ids = val_ids.union(test_ids)
        train_collection = all_annotations.filter(lambda a: (a.user in {55502856} and len(a.term) > 0
                                                             and a.term[0] in {35777351, 35777321, 35777459}
                                                             and a.image not in val_test_ids))
        val_rois = all_annotations.filter(lambda a: (a.user in {142954314}
                                                     and a.image in val_ids
                                                     and len(a.term) > 0 and a.term[0] in {154890363}))
        val_foreground = all_annotations.filter(lambda a: (a.user in {142954314}
                                                           and a.image in val_ids
                                                           and len(a.term) > 0 and a.term[0] in {154005477}))

        train_wsi_ids = list({an.image for an in all_annotations}.difference(val_test_ids))
        val_wsi_ids = list(val_ids)

        download_path = os.path.join(args.data_path, "crops-{}".format(args.tile_size))
        images = {_id: ImageInstance().fetch(_id) for _id in (train_wsi_ids + val_wsi_ids)}

        train_crops = [
            AnnotationCrop(images[annot.image], annot, download_path, args.tile_size)
            for annot in train_collection
        ]
        val_crops = [
            AnnotationCrop(images[annot.image], annot, download_path, args.tile_size)
            for annot in val_rois
        ]

        for crop in train_crops + val_crops:
            crop.download()

        np.random.seed(42)
        dataset = RemoteAnnotationTrainDataset(train_crops, seg_trans=segmentation_transform)
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.n_jobs,
            worker_init_fn=worker_init
        )

        # network
        device = torch.device(args.device)
        unet = Unet(args.init_fmaps, n_classes=1)
        unet.train()
        unet.to(device)

        optimizer = Adam(unet.parameters(), lr=args.lr)
        loss_fn = BCEWithLogitsLoss(reduction="mean")

        results = {
            "train_losses": [],
            "val_losses": [],
            "val_metrics": [],
            "save_path": []
        }

        for e in range(args.epochs):
            print("########################")
            print("        Epoch {}".format(e))
            print("########################")

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
            for i, roi in enumerate(val_crops):
                foregrounds = find_intersecting_annotations(roi.annotation, val_foreground)
                with torch.no_grad():
                    y_pred, y_true = predict_roi(
                        roi.wsi, roi, foregrounds, unet, device,
                        in_trans=transforms.ToTensor(),
                        batch_size=args.batch_size,
                        tile_size=args.tile_size,
                        overlap=args.overlap,
                        n_jobs=args.n_jobs
                    )
                val_losses[i] = metrics.log_loss(y_true.flatten(), y_pred.flatten())
                val_roc_auc[i] = metrics.roc_auc_score(y_true.flatten(), y_pred.flatten())

            print("------------------------------")
            print("Epoch {}:".format(e))
            val_loss = np.mean(val_losses)
            roc_auc = np.mean(val_roc_auc)
            print("> val_loss: {:1.5f}".format(val_loss))
            print("> roc_auc : {:1.5f}".format(roc_auc))
            print("------------------------------")

            filename = "{}_e_{}_val_{:0.4f}_roc_{:0.4f}.pth".format(datetime.now().timestamp(), e, val_loss, roc_auc)
            torch.save(unet.state_dict(), os.path.join(args.save_path, filename))

            results["val_losses"].append(val_loss)
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

    def run(self, results, batch_size=4, epochs=4, overlap=0, tile_size=512, lr=.001, init_fmaps=16):
        # import os
        # os.environ['MKL_THREADING_LAYER'] = 'GNU'
        argv = ["--host", str(self._cytomine_host),
                "--public_key", str(self._cytomine_public_key),
                "--private_key", str(self._cytomine_private_key),
                "--batch_size", str(batch_size),
                "--n_jobs", str(self._n_jobs),
                "--epochs", str(epochs),
                "--device", str(self._device),
                "--overlap", str(overlap),
                "--tile_size", str(tile_size),
                "--lr", str(lr),
                "--init_fmaps", str(init_fmaps),
                "--data_path", str(self._data_path),
                "--working_path", os.path.join(os.environ["SCRATCH"], os.environ["SLURM_JOB_ID"]),
                "--save_path", str(self._save_path)]
        for k, v in main(argv).items():
            results[k] = v


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])