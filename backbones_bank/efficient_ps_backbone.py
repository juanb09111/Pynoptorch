# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from .efficient_net import efficient_net as efficient_net

"""The first layer of each
sequence has a stride s and all others use stride 1"""

class efficient_ps_backbone(nn.Module):
    def __init__(self, in_channels, width, depth, r):
        super().__init__()

        self.resolution = r
        self.efficien_net = efficient_net(3,  width, depth, r)

        self.P4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P16 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        self.P32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(256)
        )

        

    def forward(self, x):
        
        
        x = F.interpolate(x, size=self.resolution)
        x, out_3, out_4, out_6, out_9  = self.efficien_net(x)
       
        # BOTTOM UP
        b_up1 = out_9
        b_up2 = F.interpolate(b_up1, size=out_6.shape[2:]) + out_6
        b_up3 = F.interpolate(b_up2, size=out_4.shape[2:]) + out_4
        b_up4 = F.interpolate(b_up3, size=out_3.shape[2:]) + out_3

        # TOP - BOTTOM
        t_down1 = out_3
        t_down2 = F.interpolate(t_down1, size=out_4.shape[2:]) + out_4
        t_down3 = F.interpolate(t_down2, size=out_6.shape[2:]) + out_6
        t_down4 = F.interpolate(t_down3, size=out_9.shape[2:]) + out_9

        # P
        P4 = F.leaky_relu(self.P4(b_up4 + t_down1))
        P8 = F.leaky_relu(self.P8(b_up3 + t_down2))
        P16 = F.leaky_relu(self.P16(b_up2 + t_down3))
        P32 = F.leaky_relu(self.P32(b_up1 + t_down4))
        return x, P4, P8, P16, P32

# %%

