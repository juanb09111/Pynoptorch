import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class mask_predictor(nn.Module):
    def __init__(self, in_channels, num_thing_classes):
        super().__init__()

        self.tr_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_thing_classes, kernel_size=1, stride=1) 
        
    def forward(self, x):
        
        x = self.tr_conv(x)
        x = self.conv(x)
        return x

# mask_head_output = torch.rand((2, 256,14,14))

# model = mask_predictor(256, 8)

# out = model(mask_head_output)

# print(out.shape)