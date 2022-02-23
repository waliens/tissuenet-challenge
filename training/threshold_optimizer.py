import os

import scipy
import numpy as np


class Thresholdable(object):
    def __init__(self, y_true, y_pred):
        self._indexes = np.argsort(y_pred)
        self._y_true = y_true[self._indexes].astype(np.bool)
        self._y_pred = y_pred[self._indexes]

    def range(self, x):
        index = np.searchsorted(self._y_pred, x)
        lower, upper = 0, 1
        if index > 0:
            lower = self._y_pred[index - 1]
        if index < self._y_pred.shape[0] - 1:
            upper = self._y_pred[index]
        return lower, upper

    def eval(self, x, eps=1e-8):
        gt = self._y_pred > x
        denom = 2 * np.count_nonzero(np.logical_and(gt, self._y_true))
        numer = np.count_nonzero(gt) + np.count_nonzero(self._y_true)
        return denom / (numer + eps)

    @property
    def y_pred(self):
        return self._y_pred

    def __call__(self, x):
        return self.eval(x)


def interv_overlap(interv1, interv2):
    return interv1[0] <= interv2[0] < interv1[1] or interv1[0] < interv2[1] <= interv1[1]


def gss(f: Thresholdable, a, b, tol=1e-3):
    """Golden-section search. Return the value that maximizes the function in [a, b]."""
    while not (interv_overlap(f.range(a), f.range(b)) or (b - a) > tol):
        offset = (b - a) / scipy.constants.golden
        x1, x2 = a + offset, b - offset
        f1, f2 = f(x1), f(x2)

        if f1 > f2:
            b = x2
        else:
            a = x1

    o = f.range(a)
    return (o[0] + o[1]) / 2


def linear_search(f: Thresholdable, a, b, step=0.01):
    best_th, best_score = -1, -1
    th = a
    while th <= b:
        score = f.eval(th)
        if score > best_score:
            best_score = score
            best_th = th
        th += step
    return best_th


def thresh_exhaustive_eval(f: Thresholdable, eps=1e-4):
    x = list()
    x.append(0)
    unique_preds = np.unique(f.y_pred)
    thresholds = (unique_preds[1:] + unique_preds[:-1]) / 2
    thresholds = np.hstack(([0], thresholds, [1]))
    x, y = list(), list()
    prev = -1
    for th in thresholds:
        if th - prev < eps:
            continue
        y.append(f.eval(th))
        prev = th
        x.append(th)
    return np.array(x), np.array(y)


def plot_thresh(threshs, dices, path):
    from matplotlib import pyplot as plt
    plt.figure()
    plt.ylabel("dice")
    plt.xlabel("threshold")
    plt.title(os.path.basename(path).rsplit(".", 1)[0])
    plt.plot(np.array(threshs), np.array(dices))
    plt.savefig(path)
    plt.close()

# if __name__ == "__main__":
#     np.random.seed(42)
#     y_true = np.hstack([np.ones(5), np.zeros(6)])
#     y_pred = np.random.rand(11)
#     th = Thresholdable(y_true, y_pred)
#     plot_thresh(th)
#     value = gss(th, 0, 1)
#     print(value)