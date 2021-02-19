from models_bank import mask_rcnn
# from models_bank.efficient_ps import EfficientPS
from models_bank.efficient_ps import EfficientPS
import os.path
import json
import constants
import config


def get_model():
    models_map = {
        "EfficientPS": EfficientPS(config.BACKBONE, 
            config.BACKBONE_OUT_CHANNELS, 
            config.NUM_THING_CLASSES, 
            config.NUM_STUFF_CLASSES, 
            config.ORIGINAL_INPUT_SIZE_HW,
            config.MIN_SIZE,
            config.MAX_SIZE)
    }
    return models_map[config.MODEL]