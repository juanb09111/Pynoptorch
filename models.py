from models_bank import mask_rcnn
from models_bank.efficient_ps import EfficientPS
import os.path
import json
import constants
import config



def __get_num_obj_classes():
    data_loader_train_filename_obj = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME_OBJ)

    with open(data_loader_train_filename_obj) as hasty_file:
        data = json.load(hasty_file)
        # list of categories
        categories = data["categories"]
        # get background categories
        obj_categories = list(filter(lambda x: x["supercategory"] == "object", categories))
        obj_categories_ids = list(map(lambda x: x["id"], obj_categories))
    
    return len(obj_categories_ids) + 1

def __get_num_bg_classes():
    data_loader_train_filename_obj = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME_BG)

    with open(data_loader_train_filename_obj) as hasty_file:
        data = json.load(hasty_file)
        # list of categories
        categories = data["categories"]
        # get background categories
        obj_categories = list(filter(lambda x: x["supercategory"] == "background", categories))
        obj_categories_ids = list(map(lambda x: x["id"], obj_categories))
    
    return len(obj_categories_ids) + 1



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