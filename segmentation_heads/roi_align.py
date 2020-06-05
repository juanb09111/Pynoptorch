import torch
from torch.nn import functional as F
from torch import nn, Tensor

import torchvision
from torchvision.ops import MultiScaleRoIAlign
import config
from collections import OrderedDict
from common_blocks.image_list import ImageList
from torch.jit.annotations import List, Optional, Dict, Tuple
from utils.tensorize_batch import tensorize_batch

class RoiAlign(nn.Module):

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(RoiAlign, self).__init__()
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=featmap_names,
                                                             output_size=output_size,
                                                             sampling_ratio=2)

    def forward(self, featmap_dict, boxes, image_sizes):
        x = self.roi_pooler(featmap_dict, boxes, image_sizes)
        print("roialign x. shape", x.shape)
        return x


# feature_maps = OrderedDict([('P4', torch.rand((2, 256,256,512))), ('P8', torch.rand((2, 256,128,256))), ('P16', torch.rand((2, 256,64,128))), ('P32', torch.rand((2, 256,32,64)))])

# data_loader_train = torch.load(config.DATA_LOADER_TRAIN_FILANME)
# data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

# iterator = iter(data_loader_val)

# images, anns = next(iterator)

# images = tensorize_batch(images)
# image_sizes = [x.shape[1:] for x in images]

# boxes = [torch.rand((2000, 4)), torch.rand(2000, 4)]
# print(image_sizes)

# model = RoiAlign(['P4', 'P8', 'P16', 'P32'], 14, 2)

# output = model(feature_maps, boxes, image_sizes)

# print(output.shape)