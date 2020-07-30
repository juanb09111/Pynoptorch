import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class mask_head(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.layers = nn.ModuleList()

        for _ in range(4):
            self.layers.append(depth_wise_conv(in_channels, kernel_size=3))
        
    def forward(self, x):

        for _, layer in enumerate(self.layers):
            x = layer(x)
            
        return x

# roi_output = torch.rand((2, 256,14,14))

# model = mask_head(256)

# out = model(roi_output)

# print(out.shape)