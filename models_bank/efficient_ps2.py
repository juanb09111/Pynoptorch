import constants
from utils.tensorize_batch import tensorize_batch
from segmentation_heads.sem_seg import segmentation_head as sem_seg_head
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

import config
import temp_variables
from backbones_bank.tunned_maskrcnn.mask_rcnn import MaskRCNN, maskrcnn_resnet50_fpn


class EfficientPS2(nn.Module):
    def __init__(self, backbone_net_name,
                 backbone_out_channels,
                 num_ins_classes, num_sem_classes,
                 original_image_size,
                 min_size=800, max_size=1333):
        super(EfficientPS2, self).__init__()

        self.mask_rcnn = maskrcnn_resnet50_fpn(
            pretrained=False, num_classes=num_ins_classes + 1, min_size=min_size, max_size=max_size)

        self.semantic_head = sem_seg_head(
            backbone_out_channels,
            num_ins_classes + num_sem_classes + 1, original_image_size)

        for module in self.children():
            if self.training:
                module.training = True
            else:
                module.training = False

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, images, anns=None, semantic=True, instance=True):

        losses = {}
        semantic_logits = []

        images = list(image for image in images)

        if self.training:
            maskrcnn_losses, backbone_feat = self.mask_rcnn(images, anns)
            # print("maskrcnn_losses", maskrcnn_losses)

        else:
            maskrcnn_results, backbone_feat = self.mask_rcnn(images)

        P4, P8, P16, P32 = backbone_feat['0'], backbone_feat['1'], backbone_feat['2'], backbone_feat['3']

        if semantic:
            semantic_logits = self.semantic_head(P4, P8, P16, P32)
            # print("semantic_logits.shape", semantic_logits[0, 3, :, :])

        if self.training:

            if semantic:
                semantic_masks = list(
                    map(lambda ann: ann['semantic_mask'], anns))
                semantic_masks = tensorize_batch(semantic_masks, temp_variables.DEVICE)
                target_shape = semantic_masks.long().shape 
                sem_out = F.interpolate(semantic_logits, target_shape[1:])

                losses["semantic_loss"] = F.cross_entropy(
                    sem_out, semantic_masks.long())


            losses = {**losses, **maskrcnn_losses}

            return losses

        else:
            return [{**maskrcnn_results[idx], 'semantic_logits': semantic_logits[idx]} for idx, _ in enumerate(images)]


# device = torch.device(
#     'cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Device: ", device)
# temp_variables.DEVICE = device

# train_dir = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), "..", constants.TRAIN_DIR)


# train_ann_filename = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), "..", constants.COCO_ANN_LOC, constants.ANN_TRAIN_DEFAULT_NAME)

# coco_ann_train = os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), "..", train_ann_filename)
# data_loader_train = get_datasets.get_dataloaders(
#     config.BATCH_SIZE, train_dir, annotation=coco_ann_train, semantic_masks_folder=config.SEMANTIC_SEGMENTATION_DATA)


# iterator = iter(data_loader_train)

# images, anns = next(iterator)
# images = tensorize_batch(images, device)

# # images = list(image for image in images)

# annotations = [{k: v.to(device) for k, v in t.items()}
#                for t in anns]

# model = EfficientPS(config.BACKBONE,
#                     config.BACKBONE_OUT_CHANNELS,
#                     config.NUM_THING_CLASSES,
#                     config.NUM_STUFF_CLASSES,
#                     config.ORIGINAL_INPUT_SIZE_HW)

# model.to(device)
# model.train()

# losses = model(images, anns=annotations, semantic=True, instance=True)

# print(losses)
# # model.eval()

# # with torch.no_grad():
# #     predictions = model(images)
# #     print(predictions)
