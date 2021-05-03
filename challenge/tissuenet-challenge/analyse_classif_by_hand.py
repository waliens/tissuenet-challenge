import csv
import os
import pickle
from shutil import copyfile

import numpy as np
from collections import defaultdict

from PIL import Image, ImageDraw
from numpy.random.mtrand import RandomState
from shapely.geometry import box
from shapely.affinity import affine_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skimage.color import hsv2rgb
import pyvips

from assets.inference import determine_tissue_extract_level
from svm_classifier_train import compute_challenge_score, group_per_slide


def get_results(path):
    with open("results.csv", "r") as file:
        reader = csv.DictReader(file,
                                fieldnames=["filename", "tilex", "tiley", "tilew", "tileh", "p0", "p1", "p2", "p3"],
                                delimiter=",")
        next(reader)

        per_filename = defaultdict(list)
        class_dct = defaultdict(lambda: defaultdict(lambda: 0))
        probas_dict = defaultdict(lambda: np.zeros([4], dtype=np.float))

        for row in reader:
            row_probas = np.array([float(row["p0"]), float(row["p1"]), float(row["p2"]), float(row["p3"])])
            tile = box(int(row['tilex']), int(row['tiley']), int(row['tilex']) + int(row['tilew']),
                       int(row['tiley']) + int(row['tileh']))
            per_filename[row["filename"]].append((tile, row_probas))

            pred = np.argmax(row_probas)
            class_dct[row['filename']][pred] += 1
            probas_dict[row['filename']] += row_probas

        return per_filename, class_dct, probas_dict


def get_dataset_cls_dict(slides, cls_dict, slide2cls, norm=False):
    x = np.zeros([len(slides), 4], dtype=np.int)
    y = np.zeros([len(slides)], dtype=np.int)
    for i, slide in enumerate(slides):
        x[i] = np.array([cls_dict[slide][cls] for cls in range(4)])
        y[i] = slide2cls[slide]
    if norm:
        x = x / np.sum(x, axis=1, keepdims=True)
    return x, y


def get_dataset_probas_dict(slides, probas_dict, slide2cls, norm=False):
    x = np.zeros([len(slides), 4], dtype=np.float)
    y = np.zeros([len(slides)], dtype=np.int)
    for i, slide in enumerate(slides):
        x[i] = probas_dict[slide]
        y[i] = slide2cls[slide]
    if norm:
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


def get_color_by_scale(val, minval=0, maxval=1):
    s, v = 0.45, 1.0
    min_h, max_h = 0, 0.305556
    prop = (maxval - val) / (maxval - minval)
    h = min_h + (max_h - min_h) * prop
    return tuple((hsv2rgb([h, s, v]) * 255).astype(np.uint8))


def get_color_by_class(cls):
    return [
        (180, 255, 140),
        (255, 234, 140),
        (255, 192, 140),
        (255, 148, 148)
    ][cls]


def draw_pred_on_slide(slide_path, preds, value_fn, color_fn, opacity=127, size=2048, base_zl=2):
    level = determine_tissue_extract_level(slide_path, desired_processing_size=size)
    slide = pyvips.Image.new_from_file(slide_path, page=level)

    black = (0, 0, 0, 0)
    mask = Image.new('RGBA', (slide.width, slide.height), black)
    draw = ImageDraw.Draw(mask)
    zoom_ratio = 2 ** (base_zl - level)
    t_matrix = [zoom_ratio, 0, 0, zoom_ratio, 0, 0]

    for rect, probas in preds:
        color = color_fn(value_fn(probas))
        with_opacity = color + (opacity, )
        draw.rectangle(affine_transform(rect, t_matrix).bounds, fill=with_opacity, outline=black)

    np_image = np.ndarray(
        buffer=slide.write_to_memory(),
        dtype=np.uint8,
        shape=[slide.height, slide.width, slide.bands]
    )
    pil_slide = Image.fromarray(np_image).convert("RGBA")
    img = Image.alpha_composite(pil_slide, mask)
    return pil_slide, img.convert("RGB")


def move_samples(mask, y_true, y_pred, slides, src_path, dst_folder):
    names = slides[mask]
    dst_path = os.path.join(src_path, dst_folder)
    for i, filename in enumerate(names):
        name = filename.rsplit(".", 1)[0]
        raw_filename = name + ".png"
        pred_filename = name + "_pred.png"

        dst_raw = os.path.join(dst_path, str(y_true[mask][i]), "{}_".format(y_pred[mask][i]) + raw_filename)
        dst_pred = os.path.join(dst_path, str(y_true[mask][i]), "{}_".format(y_pred[mask][i]) + pred_filename)
        os.makedirs(os.path.dirname(dst_raw), exist_ok=True)
        copyfile(os.path.join(src_path, raw_filename), dst_raw)
        copyfile(os.path.join(src_path, pred_filename), dst_pred)


def main():
    slidenames, slide2annots, slide2cls = group_per_slide("/scratch/users/rmormont/tissuenet/metadata/")

    per_filename, cls_dict, probas_dict = get_results("results.csv")

    RANDOM_SEED = 42
    TRAIN_SIZE = 0.7
    N_JOBS = 4

    random_state = np.random.RandomState(RANDOM_SEED)
    train_slides, test_slides = train_test_split(slidenames, test_size=1 - TRAIN_SIZE, random_state=random_state)

    eval_max_rank(test_slides, cls_dict, slide2cls)

    x_train, y_train = get_dataset_probas_dict(train_slides, probas_dict, slide2cls, norm=False)
    x_test, y_test = get_dataset_probas_dict(test_slides, probas_dict, slide2cls, norm=False)

    rf = RandomForestClassifier(n_estimators=1000, max_features=None, random_state=RandomState(42))
    param_grid = {"min_samples_leaf": [1, 25, 50, 100, 250, 500]}
    folder = KFold(n_splits=5)
    grid = GridSearchCV(rf, param_grid, scoring=make_scorer(compute_challenge_score), n_jobs=N_JOBS, verbose=10, refit=True, cv=folder)
    grid.fit(x_train, y_train)
    print("best_params:", grid.best_params_)
    print("best_score:", grid.best_score_)
    y_pred = grid.best_estimator_.predict(x_test)
    print_eval(y_test, y_pred)

    val_names = np.array(test_slides)

    src_path = "/scratch/users/rmormont/tissuenet/pred"
    # move_samples(y_test == y_pred, y_test, y_pred, val_names, src_path, "goodclassif")
    # move_samples(y_test != y_pred, y_test, y_pred, val_names, src_path, "missclassif")

    # rf.set_params(**grid.best_params_)
    # rf.fit(np.vstack([x_train, x_test]), np.hstack([y_train, y_test]))
    #
    # with open("random_forest.pkl", "wb+") as file:
    #     pickle.dump(rf, file)

    # base_path = "/scratch/users/rmormont/tissuenet/"
    # wsi_path = os.path.join(base_path, "wsis")
    # # entropy_path = os.path.join(base_path, "entropy")
    # pred_path = os.path.join(base_path, "pred")
    # os.makedirs(pred_path, exist_ok=True)
    # opacity = 150
    # for slidename in slidenames:
    #     print(slidename, "pred")
    #     pil_slide, pred_draw = draw_pred_on_slide(
    #         slide_path=os.path.join(wsi_path, slidename),
    #         preds=per_filename[slidename],
    #         opacity=opacity,
    #         value_fn=lambda p: np.argmax(p),
    #         color_fn=get_color_by_class,
    #         size=4096
    #     )
    #     name = slidename.split(".", 1)[0]
    #     pred_draw.save(os.path.join(pred_path, name + "_pred.png"))
    #     pil_slide.save(os.path.join(pred_path, name + ".png"))
    #
    #     print(slidename, "entro")
    #     _, entropy_draw = draw_pred_on_slide(
    #         slide_path=os.path.join(wsi_path, slidename),
    #         preds=per_filename[slidename],
    #         opacity=opacity,
    #         value_fn=lambda p: (-np.sum(p * np.log(p))),
    #         color_fn=lambda v: get_color_by_scale(v, minval=0, maxval=-np.log(0.25)),
    #         size=4096
    #     )
    #     name = slidename.split(".", 1)[0]
    #     entropy_draw.save(os.path.join(pred_path, name + "_entro.png"))

if __name__ == "__main__":
    main()