import sys
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch.nn.functional import interpolate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, fsize=3, pool=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, fsize, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, fsize, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        if pool:
            self.pool = nn.MaxPool2d((2, 2))
        else:
            self.pool = None

    def forward(self, x):
        x = self.conv_block(x)
        if self.pool:
            return self.pool(x)
        return x


class MultiScaleNetwork(nn.Module):
    def __init__(self, inplanes, zooms, n_out_blocks=3):
        super(MultiScaleNetwork, self).__init__()

        self.zooms = zooms
        self.n_out_blocks = n_out_blocks
        self.inplanes = inplanes
        max_zoom = max(zooms)
        self.outplanes = inplanes * (2 ** (max_zoom + 1))
        self.in_paths = nn.ModuleDict()
        for zoom in zooms:
            nb_blocks = max_zoom - zoom + 1
            start_planes = self.inplanes * (2 ** zoom)
            modules = OrderedDict([
                ('conv1', nn.Conv2d(3, start_planes, 3, padding=1)),
                ('bn1', nn.BatchNorm2d(start_planes)),
                ('relu1', nn.ReLU(inplace=True))
            ] + [
                ('block{}'.format(i), ConvBlock(start_planes * (2 ** i), start_planes * (2 ** (i + 1)), pool=True))
                for i in range(nb_blocks)
            ])

            self.in_paths["zoom{}".format(zoom)] = nn.Sequential(modules)

        self.merged_path = nn.Sequential(
            ConvBlock(self.outplanes * len(zooms), self.outplanes, fsize=1, pool=False),
            *[ConvBlock(self.outplanes * (2 ** i), self.outplanes * (2 ** (i + 1))) for i in range(n_out_blocks)]
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def n_features(self):
        return self.outplanes * (2 ** self.n_out_blocks)

    def forward(self, x):
        inputs = [x if z == 0 else interpolate(x, mode="nearest", scale_factor=1 / (2 ** z)) for z in self.zooms]
        branches_out = [self.in_paths["zoom{}".format(z)](i) for (i, z) in zip(inputs, self.zooms)]
        merged = torch.cat(branches_out, dim=1)
        out_fmaps = self.merged_path(merged)
        return self.max_pool(out_fmaps)



def main(argv):
    model = MultiScaleNetwork(4, [0, 1, 2])
    print(model)
    print(model.n_features())
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    np_x = np.zeros([4, 3, 1024, 1024], dtype=np.float32)
    x = torch.tensor(np_x)
    y = model.forward(x)

if __name__ == "__main__":
    main(sys.argv[1:])