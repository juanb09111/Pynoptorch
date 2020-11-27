import os.path
import sys
from ignite.engine import Events, Engine
import torch
from utils.lidar_cam_projection import *
from utils import get_lidar_dataset_2 as get_lidar_dataset
from utils.tensorize_batch import tensorize_batch

import config
import temp_variables
import constants
from utils import map_hasty
from utils import get_splits
from utils.tensorize_batch import tensorize_batch
from models_bank.fuseblock_2d_3d import FuseBlock, FuseNet
import matplotlib.pyplot as plt

def pre_process_points(lidar_fov, imPts):

    batch_imPts = []
    batch_lidar_fov = []

    for idx, points in enumerate(imPts):
        _, indices = torch.unique(points, dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1
    
        points = points[unique_indices]
        new_lidar_fov = lidar_fov[idx][unique_indices]

        batch_lidar_fov.append(new_lidar_fov)
        batch_imPts.append(points)

    return batch_lidar_fov, batch_imPts

if __name__ == "__main__":

    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    temp_variables.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()

    print(torch.cuda.memory_allocated(device=device)/1e9)
    if config.USE_PREEXISTING_DATA_LOADERS:
        data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

        # ----- Test lidar projection------

        imgs, anns, lidar_imgs, lidar_fov, imPts = next(iter(data_loader_val))
        
        # remove points that are equal
        lidar_fov, imPts = pre_process_points(lidar_fov, imPts)

        lidar_data_fov = tensorize_batch(lidar_fov, device)
        imPts = tensorize_batch(imPts, device)
        imPts= torch.round(imPts).long()
        B = len(imgs)

        # # move calib to device
        # calib = [cal.to(device) for cal in calib]
        # #move lidar data to device
        # # lidar_data = [points.to(device) for points in lidar_data]
        # lidar_data = tensorize_batch(lidar_data, device)
        
        # # the claibration matrix is the same for every sample
        # calib = calib[0]

        
        img_channel, h, w = lidar_imgs[0].shape
        # features = torch.rand((B, 256, h, w), device=device)
        # _, C, _, _ = features.shape
        
        mask = torch.zeros((B,h,w), device=temp_variables.DEVICE, dtype=torch.bool)
        sparse_depth = torch.zeros((B,1,h,w), device=temp_variables.DEVICE)
        
        # print("mask shape", mask.shape)


        # k_number = 3
        k_number = 3
        batch_k_nn_indices = find_k_nearest(k_number, lidar_data_fov)
        # print(batch_k_nn_indices.shape)

        # print(lidar_data)
        # mask, batch_lidar_fov, batch_pts_fov, batch_k_nn_indices = pre_process_points(features, lidar_data, calib, k_number)

       

        # TODO: guarantee this number is the same throughout the samples
        n_number = lidar_data_fov.shape[1] # number of samples per batch
        
        for i in range(B):
            mask[i, imPts[i, :, 1], imPts[i, :, 0]] = True

            sparse_depth[i, 0, imPts[i, :, 1], imPts[i, :, 0]] = lidar_data_fov[i, :, 2]

            print(torch.max(sparse_depth))

        plt.imshow(sparse_depth[0,0].cpu().numpy())
        plt.show()
        # tensor lidar imgs        
        lidar_imgs = tensorize_batch(lidar_imgs, device=device)

        # batched_feat = features.permute(0, 2, 3, 1)[mask].view(B, -1, C)
       
        # n_number, n_feat  = batched_feat.shape[1:]
        # show_lidar_2_img(lidar_imgs, lidar_data_fov, imPts)
        # print(lidar_imgs.shape, sparse_depth.shape, mask.shape, lidar_data_fov.shape, batch_k_nn_indices.shape)
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated(device=device)/1e9)
        model = FuseNet(k_number, n_number)
        # # model = FuseBlock(C, k_number, n_number)
        # # model = Two_D_Branch(C)
        model.to(device)
        model.eval()

        # # model(mask, features, lidar_data_fov, batch_k_nn_indices)
        out  = model(lidar_imgs, sparse_depth, mask, lidar_data_fov, batch_k_nn_indices)

        print(out.shape)
        # out = model(features)

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

        # save data_loader

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME)

        torch.save(data_loader_val, data_loader_val_filename)
        print("done")
        # ----- Test lidar projection------

        # imgs, anns, lidar_imgs, lidar_data, calib = next(iter(data_loader_val))

        # # move calib to device
        # calib = [cal.to(device) for cal in calib]
        # #move lidar data to device
        # lidar_data = [points.to(device) for points in lidar_data]

        # # the claibration matrix is the same for every sample
        # calib = calib[0]

        
        # model = Three_D_Branch(256)
        # model.to(device)
        # model.eval()

        # img_height, img_width, img_channel = lidar_imgs[0].shape

        # features = torch.rand((2, 256, img_height, img_width), device=device)

        # model(features, lidar_data, calib, lidar_imgs)
 
            
