import torch
from torch.nn import functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops import boxes as box_ops

from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
import config

class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm2d(in_channels)
        )

        self.cls_logits = nn.Sequential(
            nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_anchors)
        )
        
        self.bbox_pred = nn.Sequential(
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_anchors * 4)
        )


        for seq in self.children():
            for l in seq.children():
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        # type: (List[Tensor])
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.leaky_relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

# feature_maps = list([torch.rand((2,256,256,512)), torch.rand((2,256,128,256)), torch.rand((2,256,64,128)), torch.rand((2,256,32,64))])
# model = RPNHead(256,10)
# logits, bbox_reg = model(feature_maps)
# print(bbox_reg[1].shape)
# print(config.MAX_EPOCHS)