import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


class FocalLossWithLogits(BCEWithLogitsLoss):
    def __init__(self, gamma=1, alpha=0.5, reduction="none", **kwargs):
        super().__init__(reduction="none", **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self._floss_reduc = reduction

    def forward(self, input, target):
        bce = super().forward(input, target)
        pt = torch.exp(-bce)
        alphas = self.alpha


class BCEWithWeights(BCEWithLogitsLoss):
    def __init__(self, reduction="none", **kwargs):
        super().__init__(reduction="none", **kwargs)
        self._w_reduction = reduction

    def forward(self, input, target, weights=None):
        to_aggr = super().forward(input, target)
        if weights is not None:
            to_aggr *= weights
        if self._w_reduction == "mean":
            return torch.mean(to_aggr)
        elif self._w_reduction == "sum":
            return torch.sum(to_aggr)
        else:
            return to_aggr

class DiceWithLogitsLoss(nn.Module):
    def __init__(self, reduction="mean", smooth=1.):
        super().__init__()
        self._smooth = smooth
        self._reduction = reduction

    def forward(self, logits, y_true):
        y_pred = torch.sigmoid(logits)
        hw_dim = (-2, -1)
        intersection = torch.sum(y_true * y_pred, dim=hw_dim)
        sum_true = torch.sum(torch.square(y_true), dim=hw_dim)
        sum_pred = torch.sum(torch.square(y_pred), dim=hw_dim)
        batch_dice = 1 - (2 * intersection + self._smooth) / (sum_true + sum_pred + self._smooth)
        if self._reduction is None:
            return batch_dice
        elif self._reduction == "mean":
            return torch.mean(batch_dice)
        else:
            raise ValueError("unknown aggregation")


class MergedLoss(nn.Module):
    def __init__(self, *losses, aggr="sum", weights=None):
        super().__init__()
        self.losses = losses
        self._aggr = aggr
        if weights is None:
            weights = torch.ones([len(losses)], requires_grad=True)
        self.register_buffer('weights', weights)

    def forward(self, logits, y_true):
        if self._aggr == "sum":
            return self._accum(logits, y_true, lambda a, b: a + b)
        else:
            raise ValueError("unknown aggregation")

    def _accum(self, logits, y_true, fn):
        total_loss = self.weights[0] * self.losses[0](logits, y_true)
        for i, loss in enumerate(self.losses[1:]):
            total_loss = fn(total_loss, self.weights[i + 1] * loss(logits, y_true))
        return total_loss


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

    def forward(self, x, sigmoid=False):
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
        if sigmoid:
            x = torch.sigmoid(x)
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


# if __name__ == "__main__":
#     torch.default_generator.manual_seed(42)
#     y_true = torch.tensor([[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]], [[1., 1., 0.], [1., 1., 0.], [1., 1., 0.]]])
#     x = torch.rand(2, 3, 3)
#
#     class Custom(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.p = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
#             self.b = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
#
#         def forward(self, x):
#             return x * self.p + self.b
#
#
#     def call(loss_fn, x, y):
#         linear = Custom()
#         y_pred = linear(x)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         print(loss.detach().item())
#         print(linear.p.grad)
#         print(linear.b.grad)
#         linear.zero_grad()
#
#     call(DiceWithLogitsLoss(reduction="mean"), x, y_true)
#     call(MergedLoss(*[DiceWithLogitsLoss(reduction="mean"), BCEWithLogitsLoss(reduction="mean")]), x, y_true)
