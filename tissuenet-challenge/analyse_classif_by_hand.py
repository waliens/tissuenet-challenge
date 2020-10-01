import csv
import numpy as np
from collections import defaultdict

from numpy.random.mtrand import RandomState
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from svm_classifier_train import compute_challenge_score, group_per_slide


def get_results(path):
    with open("results.csv", "r") as file:
        reader = csv.DictReader(file,
                                fieldnames=["filename", "tilex", "tiley", "tilew", "tileh", "p0", "p1", "p2", "p3"],
                                delimiter=",")
        next(reader)

        per_filename = defaultdict(list)
        class_dct = defaultdict(lambda: defaultdict(lambda: 0))

        for row in reader:
            row_probas = np.array([float(row["p0"]), float(row["p1"]), float(row["p2"]), float(row["p3"])])
            tile = box(int(row['tilex']), int(row['tiley']), int(row['tilex']) + int(row['tilew']),
                       int(row['tiley']) + int(row['tileh']))
            per_filename[row["filename"]].append((tile, row_probas))

            pred = np.argmax(row_probas)
            class_dct[row['filename']][pred] += 1

        return per_filename, class_dct


def get_dataset_cls_dict(slides, cls_dict, slide2cls):
    x = np.zeros([len(slides), 4], dtype=np.int)
    y = np.zeros([len(slides)], dtype=np.int)
    for i, slide in enumerate(slides):
        x[i] = np.array([cls_dict[slide][cls] for cls in range(4)])
        y[i] = slide2cls[slide]
    x = x / np.sum(x, axis=1, keepdims=True)
    return x, y


def eval_max_rank(slides, cls_dict, slide2cls):
    print("eval_max_rank")
    y_val_true, y_val_pred = list(), list()
    for filename in slides:
        y_val_true.append(slide2cls[filename])
        y_val_pred.append(max(cls_dict[filename].items(), key=lambda t: t[1])[0])
    print_eval(y_val_true, y_val_pred)


def print_eval(y_true, y_pred):
    val_slide_acc = accuracy_score(y_true, y_pred)
    val_slide_score = compute_challenge_score(y_true, y_pred)
    val_slide_cm = confusion_matrix(y_true, y_pred)
    print("> slide acc: ", val_slide_acc)
    print("> slide sco: ", val_slide_score)
    print("> slide cm : ")
    print(val_slide_cm)


def main():
    slidenames, slide2annots, slide2cls = group_per_slide("/scratch/users/rmormont/tissuenet/metadata/")

    per_filename, cls_dict = get_results("results.csv")

    RANDOM_SEED = 42
    TRAIN_SIZE = 0.7
    N_JOBS = 4

    random_state = np.random.RandomState(RANDOM_SEED)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - TRAIN_SIZE, random_state=random_state)

    eval_max_rank(test_slides, cls_dict, slide2cls)

    x_train, y_train = get_dataset_cls_dict(train_slides, cls_dict, slide2cls)
    x_test, y_test = get_dataset_cls_dict(test_slides, cls_dict, slide2cls)

    rf = RandomForestClassifier(n_estimators=1000, max_features=None, random_state=RandomState(42))
    param_grid = {"min_samples_leaf": [1, 25, 50, 100, 250, 500]}
    folder = KFold(n_splits=5)
    grid = GridSearchCV(rf, param_grid, scoring=make_scorer(compute_challenge_score), n_jobs=4, verbose=10, refit=True, cv=folder)
    grid.fit(x_train, y_train)
    print("best_params:", grid.best_params_)
    print("best_score:", grid.best_score_)
    print_eval(y_test, grid.best_estimator_.predict(x_test))


if __name__ == "__main__":
    main()