import torch
from cytomine import CytomineJob
from sldc import SSLWorkflowBuilder, SemanticSegmenter
from sldc.workflow import Workflow
from sldc_cytomine import CytomineSlide, CytomineTileBuilder

from inference.postproc import PostProcessing
from inference.unet import Unet


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
        self._postproc = PostProcessing(threshold=threshold)

    def segment(self, image):
        raise NotImplementedError("not implemented")

    def segment_batch(self, images):
        with torch.no_grad():
            x = torch.from_numpy(images).to(self._device)
            pred = torch.sigmoid(self._unet(x))
            post = self._postproc(pred)
        return post.detach().cpu().numpy()


def main(argv):
    with CytomineJob.from_cli(argv) as job:
        device = "cpu"

        # network
        unet = Unet(init_depth=8, n_classes=2)
        unet.load_state_dict("file.file")
        unet.eval()

        segmenter = UNetSegmenter(
            device=device,
            unet=unet,
            classes=[0, 1],
            threshold=job.parameters.threshold
        )

        builder = CytomineTileBuilder("./tmp")
        builder = SSLWorkflowBuilder()
        builder.set_n_jobs(job.parameters.cytomine_n_jobs)
        builder.set_overlap(job.parameters.tile_overlap)
        builder.set_tile_size(job.parameters.tile_size, job.parameters.tile_size)
        builder.set_tile_builder(builder)
        builder.set_border_tiles(Workflow.BORDER_TILES_EXTEND)
        builder.set_background_class(0)
        builder.set_distance_tolerance(1)
        builder.set_seg_batch_size(job.parameters.batch_size)
        builder.set_segmenter(segmenter)
        workflow = builder.get()

        slide = CytomineSlide(
            id_img_instance=job.parameters.cytomine_id_image,
            zoom_level=job.parameters.cytomine_zoom_level
        )
        results = workflow.process(slide)



if __name__ == "__main__":
    import sys
    main(sys.argv[1:])