import pickle
from collections import defaultdict

from clustertools import build_datacube
import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from analyse_classif_by_hand import get_results, get_dataset_probas_dict, get_dataset_cls_dict, print_eval
from svm_classifier_train import group_per_slide, compute_challenge_score

MODEL_TO_LR = {
    "densenet121_imagenet_e_16_val_0.8092_sco_0.9523_z1_1602269610.556128.pth": 0.0001,
    "densenet121_imagenet_e_37_val_0.7714_sco_0.9280_z1_1602297064.95862.pth": 0.001,
    "densenet121_mtdp_e_35_val_0.8125_sco_0.9548_z1_1602294993.20278.pth": 0.0001,
    "densenet121_mtdp_e_42_val_0.7812_sco_0.9372_z1_1602302274.124153.pth": 0.001,
    "resnet18_imagenet_e_20_val_0.7936_sco_0.9400_z1_1602284618.815257.pth": 0.0001,
    "resnet18_imagenet_e_25_val_0.7656_sco_0.9342_z1_1602288221.466545.pth": 0.001,
    "resnet34_imagenet_e_29_val_0.8018_sco_0.9463_z1_1602292926.445457.pth": 0.0001,
    "resnet34_imagenet_e_39_val_0.7664_sco_0.9310_z1_1602304468.116915.pth": 0.001,
    "resnet50_imagenet_e_13_val_0.8092_sco_0.9463_z1_1602278212.460756.pth": 0.0001,
    "resnet50_imagenet_e_31_val_0.7640_sco_0.9329_z1_1602294758.970913.pth": 0.001,
    "resnet50_mtdp_e_13_val_0.8084_sco_0.9457_z1_1602278600.525486.pth": 0.0001,
    "resnet50_mtdp_e_22_val_0.7590_sco_0.9336_z1_1602288070.805371.pth": 0.001,
    "densenet121_imagenet_e_56_val_0.8067_sco_0.9482_z2_1602274272.266287.pth": 0.0001,
    "densenet121_imagenet_e_18_val_0.7985_sco_0.9426_z2_1602270033.686283.pth": 0.001,
    "densenet121_mtdp_e_13_val_0.8035_sco_0.9437_z2_1602261301.097603.pth": 0.0001,
    "densenet121_mtdp_e_20_val_0.8166_sco_0.9529_z2_1602270971.797485.pth": 0.001,
    "resnet18_imagenet_e_29_val_0.7829_sco_0.9394_z2_1602273845.577956.pth": 0.0001,
    "resnet18_imagenet_e_20_val_0.7689_sco_0.9356_z2_1602271013.427048.pth": 0.001,
    "resnet34_imagenet_e_13_val_0.7928_sco_0.9416_z2_1602268977.799415.pth": 0.0001,
    "resnet34_imagenet_e_23_val_0.7697_sco_0.9355_z2_1602272360.440732.pth": 0.001,
    "resnet50_imagenet_e_16_val_0.7919_sco_0.9454_z2_1602270120.230495.pth": 0.0001,
    "resnet50_imagenet_e_31_val_0.7582_sco_0.9326_z2_1602274314.81669.pth": 0.001,
    "resnet50_mtdp_e_42_val_0.7977_sco_0.9470_z2_1602278632.321392.pth": 0.0001,
    "resnet50_mtdp_e_27_val_0.7673_sco_0.9352_z2_1602274284.932415.pth": 0.001,
    "densenet121_imagenet_e_27_val_0.7714_sco_0.9405_z3_1602497438.170938.pth": 0.0001,
    "densenet121_imagenet_e_42_val_0.7903_sco_0.9427_z3_1602500515.90902.pth": 0.001,
    "densenet121_mtdp_e_29_val_0.7821_sco_0.9394_z3_1602497634.902743.pth": 0.0001,
    "densenet121_mtdp_e_29_val_0.7870_sco_0.9433_z3_1602497749.006002.pth": 0.001,
    "resnet18_imagenet_e_52_val_0.7599_sco_0.9376_z3_1602503546.608902.pth": 0.0001,
    "resnet18_imagenet_e_30_val_0.7401_sco_0.9293_z3_1602499053.351917.pth": 0.001,
    "resnet34_imagenet_e_49_val_0.7656_sco_0.9394_z3_1602502754.449239.pth": 0.0001,
    "resnet34_imagenet_e_16_val_0.7426_sco_0.9309_z3_1602496152.337871.pth": 0.001,
    "resnet50_imagenet_e_59_val_0.7821_sco_0.9447_z3_1602505790.992334.pth": 0.0001,
    "resnet50_imagenet_e_29_val_0.7393_sco_0.9275_z3_1602498166.004176.pth": 0.001,
    "resnet50_mtdp_e_16_val_0.7771_sco_0.9360_z3_1602495632.231674.pth": 0.0001,
    "resnet50_mtdp_e_22_val_0.7475_sco_0.9287_z3_1602496806.118761.pth": 0.001
}


def cls_dict_aggr(cls_dict, tile_pred):
    cls = np.argmax(tile_pred)
    if cls_dict is None:
        cls_dict = {i: 0 for i in range(4)}
    cls_dict[cls] += 1
    return cls_dict


def probas_sum_aggr(probas, tile_pred):
    if probas is None:
        probas = np.zeros([4], dtype=np.float)
    probas += tile_pred
    return probas


OUT_CLS_DICT = "cls_dict"
OUT_PROBAS = "probas"


def extract_tiles_individual(datacube):
    per_method = dict()
    for (model_name, ), model_cube in datacube.iter_dimensions("model_filename"):
        for param_tuple, subcube in model_cube.iter_dimensions("batch_size", "architecture", "zoom_level", "tile_size"):
            if subcube.diagnose()["Missing ratio"] > 0.0:
                continue
            _, arch, zoom, _ = param_tuple
            splitted_name = model_name.rsplit(".", 1)[0].split("_")
            pretrained = splitted_name[1]

            probas = subcube("probas")
            tile_dict = {(filename, tile): probas[i]
                          for i, (filename, tile)
                          in enumerate(zip(subcube("filenames"), subcube("tiles")))}
            per_method[(arch, pretrained, zoom, str(MODEL_TO_LR[model_name]))] = tile_dict
    return per_method


def extract_predictor_slide(datacube, tile_pred_aggr_fn=None):
    per_method_tiles = extract_tiles_individual(datacube)
    per_method_merged = dict()
    for method, tile_dict in per_method_tiles.items():
        slide_dict = dict()
        for (filename, tile), tile_pred in tile_dict.items():
            slide_dict[filename] = tile_pred_aggr_fn(slide_dict.get(filename, None), tile_pred)
        per_method_merged[method] = slide_dict
    return per_method_merged


def extract_ensemble_predictor_slide(datacube, to_merge, proba_aggr_fn=None, tile_pred_aggr_fn=None):
    if proba_aggr_fn is None:
        proba_aggr_fn = lambda f, s: f + s
    per_method_tiles = extract_tiles_individual(datacube)
    to_merge = set(to_merge)
    param_order = {k: i for i, k in enumerate(["architecture", "pretrained", "zoom_level", "lr"])}
    per_method_merged = dict()
    per_method_merged_count = defaultdict(lambda: 0)
    for method, tile_dict in per_method_tiles.items():
        key = tuple(method[i] for k, i in param_order.items() if k not in to_merge)
        merged_dict = per_method_merged.get(key, None)
        if merged_dict is None:
            per_method_merged[key] = tile_dict
        else:
            per_method_merged[key] = {k: proba_aggr_fn(p, merged_dict[k]) for k, p in tile_dict.items()}
        per_method_merged_count[key] += 1
    per_method_slide = dict()
    for method, tile_dict in per_method_merged.items():
        slide_dict = dict()
        for (filename, tile), tile_pred in tile_dict.items():
            slide_dict[filename] = tile_pred_aggr_fn(slide_dict.get(filename, None), tile_pred / per_method_merged_count[method])
        per_method_slide[method] = slide_dict
    return per_method_slide


def main():
    datacube = build_datacube("tissuenet-e2e-eval-2nd")

    print(datacube.metrics)
    print(datacube.metadata)
    print(datacube.domain)

    per_method_cls_dict = extract_predictor_slide(datacube, cls_dict_aggr)
    per_method_probas = extract_predictor_slide(datacube, probas_sum_aggr)
    per_method_cls_dict_forest = extract_ensemble_predictor_slide(datacube, to_merge=["architecture", "pretrained", "lr"], tile_pred_aggr_fn=cls_dict_aggr)
    per_method_probas_forest = extract_ensemble_predictor_slide(datacube, to_merge=["architecture", "pretrained", "lr"], tile_pred_aggr_fn=probas_sum_aggr)

    print(per_method_probas.keys())
    print(per_method_cls_dict.keys())

    slidenames, slide2annots, slide2cls = group_per_slide("/scratch/users/rmormont/tissuenet/metadata/")

    RANDOM_SEED = 42
    TRAIN_SIZE = 0.8
    N_JOBS = 16

    random_state = np.random.RandomState(RANDOM_SEED)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - TRAIN_SIZE, random_state=random_state)

    param_grid = {"min_samples_leaf": [1, 25, 50, 100, 250, 500]}
    folder = KFold(n_splits=5)

    all_methods = dict()
    for aggr_method, per_method, prepare_func in [
        ("cls_dict", per_method_cls_dict, get_dataset_cls_dict),
        ("probas", per_method_probas, get_dataset_probas_dict),
        ("cls_dict_ensemble", per_method_cls_dict_forest, get_dataset_cls_dict),
        ("probas_ensemble", per_method_probas_forest, get_dataset_probas_dict)
    ]:
        for method, x in per_method.items():
            x_train, y_train = prepare_func(train_slides, x, slide2cls, norm=False)
            x_test, y_test = prepare_func(test_slides, x, slide2cls, norm=False)
            rf = RandomForestClassifier(n_estimators=1000, max_features=None,
                                        random_state=random_state.randint(99999999))
            grid = GridSearchCV(
                rf, param_grid,
                scoring=make_scorer(compute_challenge_score),
                n_jobs=N_JOBS, refit=True, cv=folder, verbose=1)
            grid.fit(x_train, y_train)
            method_tuple = (aggr_method, ) + method
            print("-------------------------------------------------")
            print(method_tuple)
            print("best_params:", grid.best_params_)
            print("best_score:", grid.best_score_)
            y_pred = grid.best_estimator_.predict(x_test)
            print_eval(y_test, y_pred)

            all_methods[method_tuple] = (
                grid,
                compute_challenge_score(y_test, y_pred),
                grid.best_estimator_.fit(np.vstack([x_train, x_test]), np.hstack([y_train, y_test]))
            )

    best_method = None
    best_result = None
    best_score = 0
    for method, result_tuple in all_methods.items():
        grid, test_score, model = result_tuple
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_method = method
            best_result = result_tuple
        print("\t".join(map(str, method)) + "\t{}\t{}".format(grid.best_score_, test_score))

    print("Best: ", best_method)
    print("> val score : {}".format(best_result[0].best_score_))
    print("> test score: {}".format(best_result[1]))

    with open("random_forest.pkl", "wb+") as file:
        pickle.dump(best_result[2], file)


if __name__ == "__main__":
    main()