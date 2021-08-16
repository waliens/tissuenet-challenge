import os
import pickle
from collections import defaultdict
from functools import partial

from clustertools import build_datacube
import numpy as np
from clustertools.storage import PickleStorage
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from analyse_classif_by_hand import get_results, get_dataset_probas_dict, get_dataset_cls_dict, print_eval
from svm_classifier_train import group_per_slide, compute_challenge_score


def build_model_dict(experiment):
    datacube = build_datacube(experiment)
    model_dict = dict()
    base_params = ["architecture", "pretrained"]
    aug_params = list(set(datacube.parameters).difference(base_params))
    for _, cube in datacube.iter_dimensions(*base_params):
        for aug_pvalues, aug_cube in cube.iter_dimensions(*aug_params):
            if aug_cube.diagnose()["Missing ratio"] > 0.0:
                continue
            model_dict[aug_cube("models")[-1]] = [aug_cube.metadata[param] for param in aug_params]
    return aug_params, model_dict


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


def get_model_names(datacube):
    model_names = list()
    for (model_name, ), model_cube in datacube.iter_dimensions("model_filename"):
        for param_tuple, subcube in model_cube.iter_dimensions(*model_cube.parameters):
            if subcube.diagnose()["Missing ratio"] > 0.0:
                continue
            model_names.append(model_name)
    return model_names


def extract_tiles_individual(datacube, train_exp_name):
    per_method = dict()
    aug_params, model_to_params = build_model_dict(train_exp_name)
    print(aug_params)
    for (model_name, ), model_cube in datacube.iter_dimensions("model_filename"):
        for param_tuple, subcube in model_cube.iter_dimensions(*model_cube.parameters):
            if subcube.diagnose()["Missing ratio"] > 0.0:
                continue
            arch = subcube.metadata["architecture"]
            zoom = subcube.metadata["zoom_level"]
            splitted_name = model_name.rsplit(".", 1)[0].split("_")
            pretrained = splitted_name[1]

            probas = subcube("probas")
            tile_dict = {(filename, tile): probas[i]
                          for i, (filename, tile)
                          in enumerate(zip(subcube("filenames"), subcube("tiles")))}
            per_method[(arch, pretrained, zoom) + tuple(str(v) for v in model_to_params[model_name])] = tile_dict
    return per_method


def extract_predictor_slide(datacube, train_exp_name, tile_pred_aggr_fn=None):
    per_method_tiles = extract_tiles_individual(datacube, train_exp_name)
    per_method_merged = dict()
    for method, tile_dict in per_method_tiles.items():
        slide_dict = dict()
        for (filename, tile), tile_pred in tile_dict.items():
            slide_dict[filename] = tile_pred_aggr_fn(slide_dict.get(filename, None), tile_pred)
        per_method_merged[method] = slide_dict
    return per_method_merged



def one_hot(probas, dtype=np.int):
    classes = np.argmax(probas, axis=1)
    mat = np.zeros(probas.shape, dtype=dtype)
    mat[np.arange(classes.size), classes] = 1
    return mat


def aggregate_by_filename(filenames, probas, trans_fn=None, avg=False):
    if trans_fn is None:
        trans_fn = lambda i: i
    probas = trans_fn(probas)
    filenames = np.array(filenames)
    unique_f, counts = np.unique(filenames, return_counts=True)
    aggr = np.zeros([unique_f.shape[0], 4], dtype=np.float)
    for i, filename in enumerate(unique_f):
        mask = filenames == filename
        if np.any(mask):
            aggr[i] = np.sum(probas[filenames == filename], axis=0)
        else:
            aggr[i] = [1.0, 0.0, 0.0, 0.0]
    if avg:
        aggr /= counts[..., np.newaxis]
    return aggr, unique_f


def main():
    storage = PickleStorage("tissuenet-e2e-eval-3rd")
    aug_params, model_dict = build_model_dict("tissuenet-e2e-train-3rd")
    parameter_set = storage.load_parameter_set().queue
    model_names = [p["model_filename"] for p in parameter_set]

    N_JOBS = 16
    RANDOM_SEED = 42
    TRAIN_SIZE = 0.8
    SAVE_PATH = "/scratch/users/rmormont/tissuenet/trees"
    os.makedirs(SAVE_PATH, exist_ok=True)
    slidenames, slide2annots, slide2cls = group_per_slide("/scratch/users/rmormont/tissuenet/metadata/")
    random_state = np.random.RandomState(RANDOM_SEED)
    train_slides, test_slides = [set(x) for x in train_test_split(slidenames, test_size=1 - TRAIN_SIZE, random_state=random_state)]
    param_grid = {"min_samples_leaf": [1, 25, 50, 100, 250, 500]}
    folder = KFold(n_splits=5)

    print("\t".join(["arch", "pretrained", *aug_params, "reduction", "min_sample_split", "train_acc", "slide_cv", "slide_test_acc", "slide_test_score"]))
    for i, model_filename in enumerate(model_names):
        results = storage.load_result("Computation-tissuenet-e2e-eval-3rd-" + str(i))
        arch, pret = model_filename.split("_")[:2]
        val_acc = model_filename.rsplit(".", 1)[0].rsplit("_")[5]
        method = (arch, pret) + tuple(model_dict[model_filename])

        for aggr_method, aggr_params in [
            ("cls_dict", {"trans_fn": partial(one_hot, dtype=np.int), "avg": False}),
            ("probas", {"trans_fn": None, "avg": True})
        ]:
            x, order = aggregate_by_filename(results["filenames"], results["probas"], **aggr_params)
            y = np.array([slide2cls[f] for f in order])
            train = np.array([f in train_slides for f in order])
            x_train, y_train = x[train], y[train]
            x_test, y_test = x[~train], y[~train]

            rf = RandomForestClassifier(n_estimators=1000, max_features=None,
                                        random_state=random_state.randint(99999999))
            grid = GridSearchCV(
                rf, param_grid,
                scoring=make_scorer(compute_challenge_score),
                n_jobs=N_JOBS, refit=True, cv=folder, verbose=0)

            grid.fit(x_train, y_train)
            method_tuple = method + (aggr_method, )


            y_pred = grid.best_estimator_.predict(x_test)
            val_slide_acc = accuracy_score(y_test, y_pred)
            val_slide_score = compute_challenge_score(y_test, y_pred)

            print("\t".join(map(str, [
                *method_tuple,
                grid.best_params_["min_samples_leaf"],
                val_acc,
                grid.best_score_,
                val_slide_acc,
                val_slide_score
            ])))

            refit = grid.best_estimator_.fit(np.vstack([x_train, x_test]), np.hstack([y_train, y_test]))
            with open(os.path.join(SAVE_PATH, "{}.pkl".format("_".join(map(str, method)))), "wb+") as file:
                pickle.dump(refit, file)


if __name__ == "__main__":
    main()