import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from .roi_align import RoiAlign
from .box_head import box_head
from .box_predictor import box_predictor
from .mask_head import mask_head
from .mask_predictor import mask_predictor
from collections import OrderedDict
from common_blocks.image_list import ImageList
from torch.jit.annotations import List, Optional, Dict, Tuple
from utils.tensorize_batch import tensorize_batch

import config

# %%


class roi_heads(nn.Module):
    def __init__(self, num_thing_classes, backbone_out_channels=256,
                 roi_out_res=14,
                 feat_maps_names=['P4', 'P8', 'P16', 'P32'],
                 representation_size=1024,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):
        super().__init__()

        ## Boxes
        box_roi_pool = RoiAlign(feat_maps_names, roi_out_res, 2)
        bbox_head = box_head(backbone_out_channels*roi_out_res ** 2, representation_size)
        bbox_predictor = box_predictor(representation_size, num_thing_classes)

        #Masks

        mask_roi_pool = RoiAlign(feat_maps_names, roi_out_res, 2)
        m_head = mask_head(backbone_out_channels)
        m_predictor = mask_predictor(backbone_out_channels, num_thing_classes)

        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

        
        self.heads = RoIHeads(box_roi_pool, bbox_head, bbox_predictor,
                              box_fg_iou_thresh, box_bg_iou_thresh,
                              box_batch_size_per_image, box_positive_fraction,
                              bbox_reg_weights, 
                              box_score_thresh, box_nms_thresh, box_detections_per_img,
                              mask_roi_pool, m_head, m_predictor)

    def forward(self, features, proposals, image_shapes, targets=None):
        result, losses = self.heads(features, proposals, image_shapes, targets=targets)

        return result, losses


feature_maps = OrderedDict([('P4', torch.rand((2, 256,256,512))), ('P8', torch.rand((2, 256,128,256))), ('P16', torch.rand((2, 256,64,128))), ('P32', torch.rand((2, 256,32,64)))])

data_loader_train = torch.load(config.DATA_LOADER_TRAIN_FILANME)
data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

iterator = iter(data_loader_val)

images, anns = next(iterator)

images = tensorize_batch(images)
image_sizes = [x.shape[1:] for x in images]

boxes = [torch.rand((2000, 4)), torch.rand(2000, 4)]

model = roi_heads(7)
model.train()
result, losses = model(feature_maps, boxes, image_sizes, targets=anns)

# print(result, losses)

# model.eval()

# result, losses = model(feature_maps, boxes, image_sizes, targets=anns)

# print(result, losses)


