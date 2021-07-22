import torch
from torch import nn


class WeightComputer(nn.Module):
    def __init__(self, mode="constant", constant_weight=1.0, consistency_fn=None, consistency_neigh=1, logits=False, device="cpu"):
        """
        :param mode: in {'constant', 'balance_gt', 'pred_entropy', 'pred_consistency', 'pred_merged}
        :param constant_weight:
        :param consistency_fn:
        :param consistency_neigh: in pixels
        :param logits: work with logits
        """
        super().__init__()
        self._mode = mode
        self._constant_weight = constant_weight
        self._consistency_fn = consistency_fn
        self._consistency_neigh = consistency_neigh
        self._is_logits = logits
        self._device = device
        if consistency_neigh != 1 and consistency_neigh != 2:
            raise ValueError("invalid consistency neighbourhood {}".format(consistency_neigh))
        if ("consistency" in self._mode or "multi" in self._mode) and consistency_fn is None:
            raise ValueError("missing consistency function for weight computation")

    def forward(self, y, y_gt, apply_weights=None):
        weights = torch.maximum(self._weight(y, y_gt), y_gt)
        if apply_weights is not None:
            if apply_weights.ndim == 1 and apply_weights.size()[0] != weights.size()[0]:
                raise ValueError("apply weights vector does not have the correct dimensions {}".format(apply_weights.size()))
            apply_weights = apply_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1).int()
            weights = torch.maximum(weights, apply_weights)
        return weights

    def _y(self, y):
        if self._is_logits:
            return torch.sigmoid(y)
        else:
            return y

    def _weight(self, y, y_gt):
        if self._mode == "constant":
            return torch.full(y.size(), self._constant_weight, device=self._device)
        elif self._mode == "balance_gt":
            ratio = torch.mean(y_gt, dim=[2, 3], keepdim=True)
            ratio[ratio >= 1] = 0  # handle case of no background
            w = (1 - y_gt) * ratio / (1 - ratio)
            w[w > 1.0] = 1.0  # don't overweight background even if they are minority
            return w
        elif self._mode == "pred_entropy":
            return self._entropy(y)
        elif self._mode == "pred_consistency":
            return self._consistency(y)
        elif self._mode == "pred_merged":
            return self._consistency(y) * self._entropy(y)
        else:
            raise ValueError("Invalid mode '{}'".format(self._mode))

    def _entropy(self, y):
        if not self._is_logits:
            return 1 + y * torch.log2(y) + (1 - y) * torch.log2(1 - y)
        else:
            probas = torch.sigmoid(y)
            logexpy = torch.log(torch.exp(y) + 1)
            return 1 + (probas * (y - logexpy) - (1 - probas) * logexpy) / torch.log(torch.tensor(2))

    @property
    def consist_fn(self):
        if self._consistency_fn == "quadratic":
            return lambda y1, y2: torch.square(y1 - y2)
        elif self._consistency_fn == "absolute":
            return lambda y1, y2: torch.abs(y1 - y2)

    def _consistency(self, y):
        offset_range = list(range(-self._consistency_neigh, self._consistency_neigh+1))
        divider = torch.zeros(y.size(), dtype=torch.int8, device=self._device)
        accumulate = torch.zeros(y.size(), dtype=y.dtype, device=self._device)
        _, height, width = y.size()
        consist_fn = self.consist_fn
        probas = self._y(y)
        for offset_x in offset_range:
            for offset_y in offset_range:
                if offset_x == 0 and offset_y == 0:
                    continue
                ref_y_low, ref_y_high = max(0, offset_y), min(height, height + offset_y)
                ref_x_low, ref_x_high = max(0, offset_x), min(width, width + offset_x)
                tar_y_low, tar_y_high = max(0, -offset_y), min(height, height - offset_y)
                tar_x_low, tar_x_high = max(0, -offset_x), min(width, width - offset_x)

                accumulate[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high] += consist_fn(
                    probas[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high],
                    probas[:, tar_y_low:tar_y_high, tar_x_low:tar_x_high])

                divider[:, ref_y_low:ref_y_high, ref_x_low:ref_x_high] += 1

        return 1 - (accumulate / divider)
