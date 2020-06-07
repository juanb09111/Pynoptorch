import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from collections import OrderedDict
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from common_blocks.image_list import ImageList
from backbones_bank.efficient_ps_backbone import efficient_ps_backbone as eff_net
from segmentation_heads.RPN import RPN
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
from segmentation_heads.roi_heads import roi_heads
from utils.tensorize_batch import tensorize_batch
import config


class EfficientPS(nn.Module):
    def __init__(self, backbone_net_name, backbone_out_channels, num_ins_classes, num_sem_classes, original_image_size,
                 min_size=800, max_size=1333, image_mean=None, image_std=None):
        super(EfficientPS, self).__init__()

        original_aspect_ratio = original_image_size[0]/original_image_size[1]
        self.backbone = eff_net(backbone_net_name, original_aspect_ratio)
        # self.backbone = eff_net(1.6, 2.2, (512, 1024))
        # self.backbone = eff_net(1, 1, (512, 1024))

        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, original_image_size)

        self.rpn = RPN(256)
        self.roi_pool = roi_heads(num_ins_classes + 1)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def to(self, device):
        for module in self.children():
            module.to(device)

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

        roi_result, roi_losses = self.roi_pool(
            feature_maps, proposals, image_sizes, targets=anns)

        roi_result = self.transform.postprocess(
            roi_result, image_sizes, image_sizes)
            
        if self.training:
            semantic_masks = list(
                map(lambda ann: ann['semantic_mask'], anns))
            semantic_masks = tensorize_batch(semantic_masks, config.DEVICE)

            losses["semantic_loss"] = F.cross_entropy(
                semantic_logits, semantic_masks.long())

            losses = {**losses, **proposal_losses, **roi_losses}

            return losses

        else:
            # return {**roi_result, 'semantic_logits': semantic_logits}
            return [{**res, 'semantic_logits': semantic_logits[idx]} for idx, res in enumerate(roi_result)]


# device = torch.device('cpu')

# data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

# iterator = iter(data_loader_val)

# images, anns = next(iterator)

# images = tensorize_batch(images, device)

# print(images.shape[2], images.shape[3])
# ratio = images.shape[2]/images.shape[3]
# model = EfficientPS("EfficientNetB3", 256, 7, 1, (images.shape[2], images.shape[3]), ratio)

# model.to(device)
# model.train()
# losses = model(images, anns=anns)
# print(losses)

# model.eval()
# res = model(images)
# print(len(res), res[0].keys())
