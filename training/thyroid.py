import os
import joblib
from collections import defaultdict

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstance
from joblib import delayed
from shapely import wkt
from shapely.affinity import affine_transform
from shapely.geometry import box
from sldc import batch_split

from dataset import CropTrainDataset, AnnotationCrop, DatasetsGenerator

THYROID_PROJECT_ID = 77150529
VAL_IDS = {77150767, 77150761, 77150809}
TEST_IDS = {77150623, 77150611, 77150755}
EXCLUDED_WSIS = {77151057}
VAL_TEST_IDS = VAL_IDS.union(TEST_IDS)
CDEGAND_ID = 55502856
MTESTOURI_ID = 142954314

PATTERN_TERMS = {35777351, 35777321, 35777459}
CELL_TERMS = {35777375, 35777387, 35777345, 35777441, 35777393, 35777447, 35777339}
VAL_ROI_TERMS = {154890363}
VAL_FOREGROUND_TERMS = {154005477}


def get_thyroid_annotations():
    return AnnotationCollection(project=THYROID_PROJECT_ID, showWKT=True, showMeta=True, showTerm=True).fetch()


def get_val_set(annots):
    val_rois = annots.filter(lambda a: (a.user in {MTESTOURI_ID} and a.image in VAL_IDS
                                        and len(a.term) > 0 and a.term[0] in VAL_ROI_TERMS))
    val_foreground = annots.filter(lambda a: (a.user in {MTESTOURI_ID} and a.image in VAL_IDS
                                              and len(a.term) > 0 and a.term[0] in VAL_FOREGROUND_TERMS))
    return val_rois, val_foreground


def get_train_annots(annots, terms):
    return annots.filter(lambda a: (a.user in {CDEGAND_ID} and len(a.term) > 0 and a.term[0] in terms and a.image not in VAL_TEST_IDS and a.image not in EXCLUDED_WSIS))


def get_pattern_train(annots):
    return get_train_annots(annots, PATTERN_TERMS)


def get_cell_train(annots):
    return get_train_annots(annots, CELL_TERMS)


def get_polygons_intersecting_crop_roi(crop, crops):
    (x, y), width, height = crop.image_box
    bbox = box(x, y, x + width, y + height)
    return generic_match_search(bbox, crops,
                                item_fn=lambda c: c.polygon,
                                match_fn=lambda a, b: a.intersects(b))


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


def group_annot_per_image(annots):
    dd = defaultdict(list)
    for a in annots:
        dd[a.image].append(a)
    return dd


def generic_match_search(key_item, elements, item_fn, match_fn):
    return [elem for elem in elements if match_fn(key_item, item_fn(elem))]


class ThyroidDatasetGenerator(DatasetsGenerator):
    def __init__(self, data_path, tile_size, zoom_level):
        # fetch annotations (filter val/test sets + other annotations)
        all_annotations = get_thyroid_annotations()
        pattern_collec = get_pattern_train(all_annotations)
        cell_collec = get_cell_train(all_annotations)
        train_collec = pattern_collec + cell_collec
        val_rois, val_foreground = get_val_set(all_annotations)
        train_wsi_ids = list({an.image for an in all_annotations}.difference(VAL_TEST_IDS))
        val_wsi_ids = list(VAL_IDS)

        download_path = os.path.join(data_path, "crops-{}".format(tile_size))
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
        self.val_rois_to_intersect = {
            roi.id: generic_match_search(
                key_item=wkt.loads(roi.location),
                elements=[a for a in val_foreground if a.image == roi.image],
                **match_params)
            for roi in val_rois
        }
        print("done")

        self.pattern_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, tile_size,
            zoom_level=zoom_level, intersecting=intersecting[annot.id]) for annot in pattern_collec]
        self.base_cell_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, tile_size,
            zoom_level=zoom_level, intersecting=intersecting[annot.id]) for annot in cell_collec]
        self.val_crops = [AnnotationCrop(
            images[annot.image], annot, download_path, tile_size,
            zoom_level=zoom_level) for annot in val_rois]

        for crop in self.pattern_crops + self.base_cell_crops + self.val_crops:
            crop.download()

    def sets(self):
        return self.base_cell_crops, self.pattern_crops, self.val_crops, []

    def iterable_to_dataset(self, iterable, **kwargs):
        return CropTrainDataset(iterable, **kwargs)

    def val_roi_foreground(self, val_roi):
        return self.val_rois_to_intersect[val_roi.unique_identifier]
