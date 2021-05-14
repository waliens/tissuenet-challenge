from cytomine.models import AnnotationCollection

THYROID_PROJECT_ID = 77150529
VAL_IDS = {77150767, 77150761, 77150809}
TEST_IDS = {77150623, 77150611, 77150755}
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
    return annots.filter(lambda a: (a.user in {CDEGAND_ID} and len(a.term) > 0 and a.term[0] in terms and a.image not in VAL_TEST_IDS))


def get_pattern_train(annots):
    return get_train_annots(annots, PATTERN_TERMS)


def get_cell_train(annots):
    return get_train_annots(annots, CELL_TERMS)
