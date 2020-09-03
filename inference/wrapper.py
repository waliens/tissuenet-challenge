from argparse import ArgumentParser

import os
from datetime import datetime

import numpy as np
from pathlib import Path

import torch
from clustertools import Computation
from clustertools.storage import PickleStorage
from cytomine import CytomineJob
from cytomine.models import AnnotationCollection, Annotation, ImageInstance
from shapely import wkt
from shapely.affinity import affine_transform, translate
from sldc import SSLWorkflowBuilder, SemanticSegmenter
from sldc.workflow import Workflow
from sldc_cytomine import CytomineSlide, CytomineTileBuilder

from unet import Unet


class UNetSegmenter(SemanticSegmenter):
    def __init__(self, device, unet, classes=None, threshold=0.5):
        """
        Parameters
        ----------
        device: str
        unet: nn.Module
            In eval mode,
        classes:
        """
        super().__init__(classes=classes)
        self._device = torch.device(device)
        self._unet = unet
        self._threshold = threshold

    def segment(self, image):
        raise NotImplementedError("not implemented")

    def segment_batch(self, images):
        import cv2
        filenames = []
        for i in range(images.shape[0]):
            filenames.append(os.path.join("D:\\Git", "weakseg", "inference", "images", datetime.utcnow().isoformat().replace("-", "").replace(":", "").replace(".", "") + ".png"))
            cv2.imwrite(filenames[i], images[i])
        with torch.no_grad():
            x_np = np.moveaxis(images[::-1] / 255, [3], [1]).astype(np.float32)
            x = torch.from_numpy(x_np).to(self._device)
            pred = self._unet(x, sigmoid=True)
            thresholded = pred > self._threshold
            return thresholded.detach().cpu().numpy().astype(np.uint8).squeeze(axis=1)


def shift_poly(p, inst, offset=(0, 0)):
    p = translate(p, xoff=offset[0], yoff=offset[1])
    p = affine_transform(p, [1, 0, 0, -1, 0, inst.height])
    return p


def main(argv):
    with CytomineJob.from_cli(argv) as job:
        # network
        device = torch.device(job.parameters.device)
        unet = Unet(job.parameters.init_fmaps, n_classes=1)
        unet.load_state_dict(torch.load("/home/rmormont/models/thyroid-unet/1599132006.366569_e_4_val_0.0402_roc_0.9957_z0.pth"))
        unet.to(device)
        unet.eval()

        segmenter = UNetSegmenter(device=job.parameters.device, unet=unet, classes=[0, 1], threshold=job.parameters.threshold)

        working_path = os.path.join(str(Path.home()), "tmp")
        tile_builder = CytomineTileBuilder(working_path)
        builder = SSLWorkflowBuilder()
        builder.set_n_jobs(job.parameters.n_jobs)
        builder.set_overlap(job.parameters.tile_overlap)
        builder.set_tile_size(job.parameters.tile_size, job.parameters.tile_size)
        builder.set_tile_builder(tile_builder)
        builder.set_border_tiles(Workflow.BORDER_TILES_EXTEND)
        builder.set_background_class(0)
        builder.set_distance_tolerance(1)
        builder.set_seg_batch_size(job.parameters.batch_size)
        builder.set_segmenter(segmenter)
        workflow = builder.get()

        slide = CytomineSlide(
            img_instance=ImageInstance().fetch(job.parameters.cytomine_id_image),
            zoom_level=job.parameters.cytomine_zoom_level
        )
        image_instance = slide.image_instance

        collection = AnnotationCollection(showWKT=True, showMeta=True).fetch_with_filter("project", 77150529)
        rois = [shift_poly(wkt.loads(annot.wkt), image_instance)
                for annot in collection
                if annot.image == job.parameters.cytomine_id_image
                    and len(annot.term) > 0 and annot.term[0] in {154890363} ]

        collection = AnnotationCollection()
        for roi in rois:
            xmin, ymin, xmax, ymax = [int(v) for v in roi.bounds]
            window = slide.window((xmin, ymax), xmax - xmin, ymax - ymin)
            results = workflow.process(window)
            for obj in results:
                collection.append(Annotation(
                    location=shift_poly(obj.polygon, image_instance, offset=window.abs_offset),
                    id_image=job.parameters.cytomine_id_image,
                    id_terms=[154005477],
                    id_project=job.project.id
                ))
        collection.save(n_workers=job.parameters.n_jobs)


class ProcessWSIComputation(Computation):
    def __init__(self, exp_name, comp_name, host=None, private_key=None, public_key=None, n_jobs=1, device="cuda:0",
                 software_id=None, project_id=None, save_path=None, data_path=None, context="n/a", storage_factory=PickleStorage, **kwargs):
        super().__init__(exp_name, comp_name, context=context, storage_factory=storage_factory)
        self._n_jobs = n_jobs
        self._device = device
        self._save_path = save_path
        self._cytomine_private_key = private_key
        self._cytomine_public_key = public_key
        self._cytomine_host = host
        self._data_path = data_path
        self._software_id = software_id
        self._project_id = project_id

    def run(self, results, image_id=None, batch_size=4, tile_size=512, tile_overlap=0, init_fmaps=16, zoom_level=0):
        argv = ["--cytomine_host", str(self._cytomine_host),
                "--cytomine_public_key", str(self._cytomine_public_key),
                "--cytomine_private_key", str(self._cytomine_private_key),
                "--software_id", str(self._software_id),
                "--project_id", str(self._project_id),
                "--cytomine_id_image", str(image_id),
                "--cytomine_zoom_level", str(0),
                "--cytomine_tile_size", str(tile_size),
                "--cytomine_tile_overlap", str(tile_overlap),
                "--n_jobs", str(self._n_jobs),
                "--batch_size", str(batch_size),
                "--device", str("cuda:0"),
                "--zoom_level", str(zoom_level)]
        for k, v in main(argv).items():
            results[k] = v


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])