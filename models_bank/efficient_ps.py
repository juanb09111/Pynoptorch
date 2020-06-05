import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from common_blocks.image_list import ImageList
from backbones_bank.efficient_ps_backbone import efficient_ps_backbone as eff_net
from segmentation_heads.RPN import RPN
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
from segmentation_heads.roi_heads import roi_heads
from utils.tensorize_batch import tensorize_batch
import config


class EfficientPS(nn.Module):
    def __init__(self, backbone_out_channels, num_ins_classes, num_sem_classes):
        super().__init__()

        self.backbone = eff_net(1.6, 2.2, (512, 1024))

        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, (1200, 1920))

        self.rpn = RPN(256)
        self.roi_pool = roi_heads(num_ins_classes + 1)

        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def forward(self, images, anns=None):

        losses = {}

        _, P4, P8, P16, P32 = self.backbone(images)
        semantic_logits = self.semantic_head(P4, P8, P16, P32)

        feature_maps = OrderedDict([('P4', P4),
                                    ('P8', P8),
                                    ('P16', P16),
                                    ('P32', P32)])
                                    
        image_list = ImageList(images, [x.shape[1:] for x in images])
        image_sizes = [x.shape[1:] for x in images]

        proposals, proposal_losses = self.rpn(
            image_list, feature_maps, anns)
        
        roi_result, roi_losses = self.roi_pool(feature_maps, proposals, image_sizes, targets=anns)

        if self.training:
            semantic_masks = list(
                map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(semantic_masks)

            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            losses["rpn"] = proposal_losses

            losses['roi'] = roi_losses

            return losses

        else:
            return semantic_logits, roi_result


# data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

# iterator = iter(data_loader_val)

# images, anns = next(iterator)

# images = tensorize_batch(images)

# model = EfficientPS(256, 7, 1)
# model.train()
# losses = model(images, anns=anns)
# print(losses)

# model.eval()
# semantic_logits, roi_result = model(images)


