import torch
from torch import nn


class WeightComputer(nn.Module):
    def __init__(self, mode="constant", constant_weight=1.0, consistency_fn=None, consistency_neigh=1):
        """
        :param mode: in {'constant', 'balance_gt', 'pred_entropy', 'pred_consistency', 'pred_merged}
        :param constant_weight:
        :param consistency_fn:
        :param consistency_neigh: in pixels
        """
        super().__init__()
        self._mode = mode
        self._constant_weight = constant_weight
        self._consistency_fn = consistency_fn
        self._consistency_neigh = consistency_neigh
        if consistency_neigh != 1 and consistency_neigh != 2:
            raise ValueError("invalid consistency neighbourhood {}".format(consistency_neigh))
        if ("consistency" in self._mode or "multi" in self._mode) and consistency_fn is None:
            raise ValueError("missing consistency function for weight computation")

    def forward(self, y, y_gt):
        return torch.maximum(self._weight(y, y_gt), y_gt)

    def _weight(self, y, y_gt):
        if self._mode == "constant":
            return torch.full(y.size(), self._constant_weight, requires_grad=True)
        elif self._mode == "balance_gt":
            ratio = torch.mean(y_gt, dim=[1, 2], keepdim=True)
            return (1 - y_gt) * ratio / (1 - ratio)
        elif self._mode == "pred_entropy":
            return self._entropy(y)
        elif self._mode == "pred_consistency":
            return 1 - self._consistency(y)
        elif self._mode == "pred_merged":
            return (1 - self._consistency(y)) * self._entropy(y)
        else:
            raise ValueError("Invalid mode '{}'".format(self._mode))

    def _entropy(self, y):
        return 1 + y * torch.log(y) + (1 - y) * torch.log(1 - y)

    @property
    def consist_fn(self):
        if self._consistency_fn == "quadratic":
            return lambda y1, y2: torch.square(y1 - y2)
        elif self._consistency_fn == "absolute":
            return lambda y1, y2: torch.abs(y1 - y2)

    def _consistency(self, y):
        offset_range = list(range(-self._consistency_neigh, self._consistency_neigh+1))
        divider = torch.zeros(y.size(), dtype=torch.int8)
        accumulate = torch.zeros(y.size(), dtype=y.dtype)
        _, height, width = y.size()
        consist_fn = self.consist_fn
        for offset_x in offset_range:
            for offset_y in offset_range:
                if offset_x == 0 and offset_y == 0:
                    continue
                ref_y_low, ref_y_high = max(0, offset_y), min(height, height + offset_y)
                ref_x_low, ref_x_high = max(0, offset_x), min(width, width + offset_x)
                tar_y_low, tar_y_high = max(0, -offset_y), min(height, height - offset_y)
                tar_x_low, tar_x_high = max(0, -offset_x), min(width, width - offset_x)

                accumulate[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high] += consist_fn(
                    y[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high],
                    y[:, tar_y_low:tar_y_high, tar_x_low:tar_x_high])

                divider[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high] += 1

        return accumulate / divider


if __name__ == "__main__":
    comp = WeightComputer(mode="pred_entropy")
    gt = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    pred = torch.tensor([[[0.75, 0.75, 0.75], [0.75, 0.75, 0.75], [0.75, 0.75, 0.75]]])
    weights = comp(pred, gt)
    print(weights.numpy())