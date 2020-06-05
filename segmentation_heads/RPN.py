from collections import OrderedDict
import torchvision
import torch
from torch.nn import functional as F
from torch import nn, Tensor
from torchvision.models.detection.rpn import RegionProposalNetwork
from .rpn_head import RPNHead
from .anchor_generator import AnchorGenerator
from common_blocks.image_list import ImageList
from torch.jit.annotations import List, Optional, Dict, Tuple
import config
from utils.tensorize_batch import tensorize_batch


class RPN(nn.Module):

    def __init__(self, backbone_out_channels, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, rpn_nms_thresh=0.7,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000):
        super(RPN, self).__init__()

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_head = RPNHead(backbone_out_channels,
                           rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                         rpn_batch_size_per_image, rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

    def forward(self, images, features, targets=None):

        if self.training:
            print('rpn training mode')
        else:
            print('rpn eval mode')
        proposals, proposal_losses = self.rpn(images, features, targets)
        return proposals, proposal_losses


# feature_maps = OrderedDict([('P4', torch.rand((2, 256, 256, 512))),
#                             ('P8', torch.rand((2, 256, 128, 256))),
#                             ('P16', torch.rand((2, 256, 64, 128))),
#                             ('P32', torch.rand((2, 256, 32, 64)))])

# data_loader_train = torch.load(config.DATA_LOADER_TRAIN_FILANME)
# data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

# iterator = iter(data_loader_val)

# images, anns = next(iterator)

# images = tensorize_batch(images)
# image_list = ImageList(images, [x.shape[1:] for x in images])
# model = RPN(256)
# model.eval()
# proposals, proposal_losses = model(image_list, feature_maps)
# print(proposals)
# print(proposals[0].shape, proposal_losses['loss_objectness'])
