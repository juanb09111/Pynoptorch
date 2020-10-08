import os.path
import sys
from ignite.engine import Events, Engine
import torch
from utils import get_lidar_dataset
from utils.tensorize_batch import tensorize_batch

import config
import temp_variables
import constants
from utils import map_hasty
from utils import get_splits
from models_bank.fuseblock_2d_3d import Three_D_Branch


if __name__ == "__main__":

    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    temp_variables.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()


    if config.USE_PREEXISTING_DATA_LOADERS:
        data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

        # ----- Test lidar projection------

        imgs, anns = next(iter(data_loader_val))
        model = Three_D_Branch(256)
        model.to(device)
        model.eval()


        lidar_img = anns[0]["lidar_img"]

        (img_height, img_width) = lidar_img.shape[1:]
        features = torch.rand((2, 256, img_height, img_width), device=device)

        lidar = [ann["lidar_data"] for ann in anns]
        lidar = torch.tensor(lidar, device=device, dtype=torch.float)

        calib = anns[0]["calib"]

        out = model(features, lidar, calib, lidar_img)

    else:
        if config.AUTOMATICALLY_SPLIT_SETS:
            get_splits.get_splits()


        val_ann_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME)

        val_ann_filename = val_ann_filename if config.COCO_ANN_VAL is None else config.COCO_ANN_VAL
        # overwrite config

        # write annotations json files for every split

        map_hasty.get_split(constants.VAL_DIR, val_ann_filename)

        val_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.VAL_DIR)


        coco_ann_val = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), val_ann_filename)


        # data loaders

        data_loader_val = get_lidar_dataset.get_dataloaders(
            config.BATCH_SIZE, val_dir, config.LIDAR_DATA, annotation=coco_ann_val, semantic_masks_folder=config.SEMANTIC_SEGMENTATION_DATA)

        # ----- Test lidar projection------

        imgs, anns, lidar_imgs, lidar_data, calib = next(iter(data_loader_val))

        # move calib to device
        calib = [cal.to(device) for cal in calib]
        #move lidar data to device
        lidar_data = [points.to(device) for points in lidar_data]

        # the claibration matrix is the same for every sample
        calib = calib[0]

        
        model = Three_D_Branch(256)
        model.to(device)
        model.eval()

        img_height, img_width, img_channel = lidar_imgs[0].shape

        features = torch.rand((2, 256, img_height, img_width), device=device)

        model(features, lidar_data, calib, lidar_imgs)
 
            
