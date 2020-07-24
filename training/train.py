import os
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
from cytomine import Cytomine
from cytomine.models import AnnotationCollection
from shapely import wkt
from sklearn import metrics
from sldc_cytomine import CytomineSlide
from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from training.dataset import RemoteAnnotationTrainDataset, segmentation_transform, predict_roi
from training.unet import Unet


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
        parser.add_argument("-w", "--working_path", dest="working_path",
                            default=os.path.join(str(Path.home()), "tmp"))
        args, _ = parser.parse_known_args(argv)

        os.makedirs(args.working_path, exist_ok=True)

        # fetch annotations (filter val/test sets + other annotations)
        all_annotations = AnnotationCollection(project=77150529, showWKT=True, showMeta=True, showTerm=True).fetch()
        train_collection = all_annotations.filter(lambda a: (a.user in {55502856} and len(a.term) > 0
                                                             and a.term[0] in {35777351, 35777321, 35777459}
                                                             and a.image not in {77150767, 77150761, 77150809, 77150623, 77150611, 77150755}))
        val_rois = all_annotations.filter(lambda a: (a.user in {142954314}
                                                     and a.image in {77150767, 77150761, 77150809}
                                                     and len(a.term) > 0 and a.term[0] in {154890363}))
        val_foreground = all_annotations.filter(lambda a: (a.user in {142954314}
                                                           and a.image in {77150767, 77150761, 77150809}
                                                           and len(a.term) > 0 and a.term[0] in {154005477}))

        val_images = {_id: CytomineSlide(_id) for _id in {r.image for r in val_rois}}
        np.random.seed(42)
        dataset = RemoteAnnotationTrainDataset(
            train_collection,
            in_trans=transforms.Lambda(torange0_1),
            seg_trans=segmentation_transform,
            working_path=args.working_path,
            cyto_argv=argv,
            width=args.tile_size,
            height=args.tile_size
        )
        loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.n_jobs
        )

        # network
        device = torch.device(args.device)
        unet = Unet(args.init_fmaps, n_classes=1)
        unet.train()
        unet.to(device)

        optimizer = Adam(unet.parameters(), lr=args.lr)
        loss_fn = BCEWithLogitsLoss(reduction="mean")

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
                sys.stdout.flush()

            unet.eval()
            # validation
            val_losses = np.zeros(len(val_rois), dtype=np.float)
            val_roc_auc = np.zeros(len(val_rois), dtype=np.float)
            for i, roi in enumerate(val_rois):
                foregrounds = find_intersecting_annotations(roi, val_foreground)
                with torch.no_grad():
                    y_pred, y_true = predict_roi(
                        val_images[roi.image], roi, foregrounds, unet, device,
                        in_trans=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Lambda(torange0_1)
                        ]),
                        batch_size=args.batch_size,
                        tile_size=args.tile_size,
                        overlap=args.overlap,
                        n_jobs=args.n_jobs,
                        working_path=args.working_path,
                        cyto_argv=argv
                    )
                val_losses[i] = metrics.log_loss(y_true.flatten(), y_pred.flatten())
                val_roc_auc[i] = metrics.roc_auc_score(y_true.flatten(), y_pred.flatten())

            print("------------------------------")
            print("Epoch {}:".format(e))
            print("> val_loss: {:1.5f}".format(np.mean(val_losses)))
            print("> roc_auc : {:1.5f}".format(np.mean(val_roc_auc)))
            print("------------------------------")



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])