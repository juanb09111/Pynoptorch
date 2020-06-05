import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.depth_wise_conv import depth_wise_conv
#%%
class box_predictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.cls_score = nn.Linear(in_channels, num_classes + 1)

        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
