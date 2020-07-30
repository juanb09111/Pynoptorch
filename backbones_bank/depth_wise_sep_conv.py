# %%
import torch
from torch import nn
import torch.nn.functional as F


class depth_wise_sep_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x