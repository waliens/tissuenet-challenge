import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import math


def dice_(y_pred, y):
    """
    compute the Dice coefficient
    (might not be very accurate)

    parameters
    ----------
    y_pred: tensor
        predictions
    y: tensor
        targets
    c_weights: float array
        the class weights
    returns
    -------
    dice: float
        dice coefficient
    """

    smooth = 1.
    y_pred = torch.sigmoid(y_pred).view(len(y_pred), -1)
    y = y.view(len(y), -1)
    intersection = torch.sum(y_pred * y)
    sum_a = torch.sum(y_pred * y_pred)
    sum_b = torch.sum(y * y)
    return ((2. * intersection + smooth) / (sum_a + sum_b + smooth))


def dice(y_pred, y, c_weights=None):
    """
    compute the Dice coefficient

    parameters
    ----------
    y_pred: tensor
        predictions tensor of shape: (batch_size, n_channels, height, width)
    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)
    c_weights: float array
        the class weights
    returns
    -------
    dice: float
        dice coefficient
    """

    sum_dice = 0
    for i in range(y.shape[0]):
        im_a = y_pred[i].unsqueeze(0)
        im_b = y[i].unsqueeze(0)
        jacc = jaccard(im_a, im_b, c_weights)
        sum_dice += ((2 * jacc) / (1 + jacc))
    return sum_dice / y.shape[0]


def jaccard(y_pred, y, c_weights=None):
    """
    compute the Jaccard index

    parameters
    ----------
    y_pred: tensor
        predictions tensor of shape: (batch_size, n_channels, height, width)
    y: tensor
        targets tensor of shape: (batch_size, n_channels, height, width)
    c_weights: float array
        class weights
    returns
    -------
    jaccard: float
        jaccard index
    """

    if c_weights is None:
        c_weights = torch.softmax(torch.ones(y.shape[1]), dim=0)
    elif len(c_weights) != y.shape[1]:
        raise ValueError("number of weights must be equal to the number of classes")
    elif torch.sum(c_weights) != 1:
        c_weights = torch.softmax(c_weights)

    sum_jacc = 0
    for i in range(y.shape[0]):
        im_a = torch.round(torch.sigmoid(y_pred[i]))
        im_b = torch.round(torch.sigmoid(y[i]))

        jacc = 0
        for j in range(y.shape[1]):
            a = im_a[j, :, :]
            b = im_b[j, :, :]
            intersection = torch.relu(a + b - 1)
            union = torch.ceil((a + b) / 2)
            jacc += ((torch.sum(intersection) / torch.sum(union)) * c_weights[j])

        sum_jacc += jacc
    return sum_jacc / y.shape[0]


def dice_loss(y_pred, y, c_weights=None):
    return 1 - dice(y_pred, y, c_weights)


class SegLoss(nn.Module):
    """
    segmentation loss : BCE + Dice
    parameters
    ----------
    c_weights: float array
        class weights used for loss computation
    """

    def __init__(self, c_weights=None):
        super().__init__()
        self._bce_loss = nn.BCEWithLogitsLoss()
        self._c_weights = c_weights

    def forward(self, y_pred, y):
        return self._bce_loss(y_pred, y) + dice_loss(y_pred, y, self._c_weights)


class Unet(nn.Module):
    def __init__(self, init_depth, n_classes):
        super().__init__()
        # encoder
        in_ch = 3
        out_ch = init_depth
        self.conv1 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch * 2
        self.conv2 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch * 2
        self.conv3 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch * 2
        self.conv4 = ConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = out_ch * 2
        self.conv5 = ConvBlock(in_ch, out_ch, pool=False)

        # decoder
        in_ch = out_ch
        out_ch = int(out_ch / 2)
        self.up_conv6 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch / 2)
        self.up_conv7 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch / 2)
        self.up_conv8 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        out_ch = int(out_ch / 2)
        self.up_conv9 = UpConvBlock(in_ch, out_ch)
        in_ch = out_ch
        self.conv10 = nn.Conv2d(in_ch, n_classes, 1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # decoder
        x = self.up_conv6(x, self.conv4.skip_x)
        x = self.up_conv7(x, self.conv3.skip_x)
        x = self.up_conv8(x, self.conv2.skip_x)
        x = self.up_conv9(x, self.conv1.skip_x)
        x = self.conv10(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.skip_x = torch.Tensor()

        if pool:
            self.pool = nn.MaxPool2d((2, 2))
        else:
            self.pool = None

    def forward(self, x):
        self.skip_x = self.conv_block(x)
        if self.pool:
            return self.pool(self.skip_x)
        return self.skip_x


class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, skip_x):
        x = self.up(x)

        # crop if necessary
        if x.shape != skip_x.shape:
            skip_x = skip_x[:, :, :x.shape[2], :x.shape[3]]

        x = torch.cat([x, skip_x], dim=1)
        return F.relu(self.conv2(F.relu(self.conv1(x))))


