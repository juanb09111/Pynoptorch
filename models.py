from models_bank import mask_rcnn
from models_bank.efficient_ps import EfficientPS
import os.path
import json
import constants
import config


def get_model():
    models_map = {
        "MaskRCNN": mask_rcnn.get_model(config.NUM_CLASSES),
        "EfficientPS": EfficientPS(config.BACKBONE, 
            config.BACKBONE_OUT_CHANNELS, 
            config.NUM_THING_CLASSES, 
            config.NUM_STUFF_CLASSES, 
            config.ORIGINAL_INPUT_SIZE_HW)
        # "DeepLab": deeplab_resnet.get_model(__get_num_bg_classes())
    }
    return models_map[config.MODEL]