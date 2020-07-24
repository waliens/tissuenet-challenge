import torch
from torch import nn


class Dilation(nn.Module):
    def __init__(self, n_iters=2, k_size=10, padding=0):
        super().__init__()
        self._n_iters = n_iters
        self._pooling = nn.MaxPool2d(k_size, stride=1, padding=padding)

    def forward(self, x):
        for i in range(self._n_iters):
            x = self._pooling(x)
        return x


class Erosion(Dilation):
    def __init__(self, n_iters=2, k_size=10, padding=1):
        super().__init__(n_iters=n_iters, k_size=k_size, padding=padding)

    def forward(self, x):
        return -super().forward(-x)


class MorphologicalOpening(nn.Module):
    def __init__(self, n_iters=2, k_size=10):
        super().__init__()
        self._erosion = Erosion(n_iters=n_iters, k_size=k_size)
        self._dilation = Dilation(n_iters=n_iters, k_size=k_size)

    def forward(self, x):
        return self._dilation(self._erosion(x))


class Normalize(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self._eps = eps

    def forward(self, x):
        return (x - torch.mean(x, dim=(1, 2), keepdim=True)) / (torch.std(x, dim=(1, 2)) + self._eps)


class PostProcessing(nn.Module):
    def __init__(self, threshold=0.5, n_iters=2, k_size=10):
        super().__init__()
        self._opening = MorphologicalOpening(n_iters=n_iters, k_size=k_size)
        self._blur = nn.AvgPool2d(k_size, stride=1)
        self._thresh = nn.Threshold(threshold=threshold, value=0, inplace=False)
        self._norm = Normalize()

    def forward(self, x):
        x = self._norm(x)
        x = self._opening(x)
        x = self._blur(x)
        return self._thresh(x)

