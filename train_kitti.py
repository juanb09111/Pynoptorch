# %%
import os.path
import sys
from ignite.engine import Events, Engine
# from ignite.contrib.handlers.param_scheduler import PiecewiseLinear
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from utils.get_kitti_dataset import get_dataloaders
from utils.tensorize_batch import tensorize_batch
from eval_coco import evaluate
from torch.utils.tensorboard import SummaryWriter


import config_kitti
import temp_variables
import constants
import models
from utils import map_hasty
from utils import get_splits


# %%
writer = SummaryWriter()

def __update_model(trainer_engine, batch):
    model.train()
    optimizer.zero_grad()

    imgs, lidar_fov, masks, sparse_depth, k_nn_indices, sparse_depth_gt = batch
    # imgs, annotations = batch[0], batch[1]
    

    imgs = list(img for img in imgs)
    lidar_fov = list(lid_fov for lid_fov in lidar_fov)
    masks = list(mask for mask in masks)
    sparse_depth = list(sd for sd in sparse_depth)
    k_nn_indices = list(k_nn for k_nn in k_nn_indices)
    sparse_depth_gt = list(sp_d for sp_d in sparse_depth_gt)
    
    
    imgs = tensorize_batch(imgs, device)
    lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
    masks = tensorize_batch(masks, device, dtype=torch.bool)
    sparse_depth = tensorize_batch(sparse_depth, device)
    k_nn_indices = tensorize_batch(k_nn_indices, device, dtype=torch.long)
    sparse_depth_gt = tensorize_batch(sparse_depth_gt, device, dtype=torch.float)
    
    
    loss_dict = model(imgs, sparse_depth, masks, lidar_fov, k_nn_indices, sparse_depth_gt)
    
    losses = sum(loss for loss in loss_dict.values())

    i = trainer_engine.state.iteration
    writer.add_scalar("Loss/train/iteration", losses, i)
    
    losses.backward()

    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

    

    optimizer.step()

    
    

    return losses
    


# %% Define Event listeners


def __log_training_loss(trainer_engine):

    current_lr = None 

    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
        

    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    i = trainer_engine.state.iteration
    text = "Epoch {}/{} : {} - batch loss: {:.2f}, LR:{}".format(
        state_epoch, max_epochs, i, batch_loss, current_lr)

    sys.stdout = open(train_res_file, 'a+')
    print(text)

    writer.add_scalar("LR/train/iteration", current_lr, i)

    


def __log_validation_results(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    weights_path = "{}{}_loss_{}.pth".format(
        constants.MODELS_LOC, config_kitti.MODEL_WEIGHTS_FILENAME_PREFIX, batch_loss)
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)

    sys.stdout = open(train_res_file, 'a+')
    print("Model weights filename: ", weights_path)
    text = "Validation Results - Epoch {}/{} batch_loss: {:.2f}".format(
        state_epoch, max_epochs, batch_loss)
    sys.stdout = open(train_res_file, 'a+')
    print(text)

    writer.add_scalar("Loss/train/epoch", batch_loss, state_epoch)

    scheduler.step()

    torch.cuda.empty_cache()


if __name__ == "__main__":

    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.RES_LOC, constants.KITTI_TRAIN_RES_FILENAME)

    with open(train_res_file, "w+") as training_results:
        training_results.write(
            "----- TRAINING RESULTS - Kitti Depth Completion----"+"\n")
    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    temp_variables.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model_by_name(config_kitti.MODEL)
    # move model to the right device
    model.to(device)

    print(torch.cuda.memory_allocated(device=device))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]

    
    # optimizer = torch.optim.SGD(
    #     params, lr=0.005, momentum=0.9, weight_decay=0.00005)

    optimizer = torch.optim.SGD(
        params, lr=0.0016, momentum=0.9, weight_decay=0.00005)
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    data_loader_train = None
    data_loader_val = None

    if config_kitti.USE_PREEXISTING_DATA_LOADERS:
        data_loader_train = torch.load(config_kitti.DATA_LOADER_TRAIN_FILANME)
        data_loader_val = torch.load(config_kitti.DATA_LOADER_VAL_FILENAME)

    else:

        imgs_root_val = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "imgs/2011_09_26/val/")
        data_depth_velodyne_root_val = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "data_depth_velodyne/val/")
        data_depth_annotated_root_val = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "data_depth_annotated/val/")

        imgs_root_train = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "imgs/2011_09_26/train/")
        data_depth_velodyne_root_train = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "data_depth_velodyne/train/")
        data_depth_annotated_root_train = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "data_depth_annotated/train/")

        calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "imgs/2011_09_26/calib_velo_to_cam.txt")
        calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), config_kitti.DATA, "imgs/2011_09_26/calib_cam_to_cam.txt")

        # data loaders
        data_loader_train = get_dataloaders(batch_size=config_kitti.BATCH_SIZE,
                                            imgs_root=imgs_root_train,
                                            data_depth_velodyne_root=data_depth_velodyne_root_train,
                                            data_depth_annotated_root=data_depth_annotated_root_train,
                                            calib_velo2cam=calib_velo2cam,
                                            calib_cam2cam=calib_cam2cam)

        data_loader_val = get_dataloaders(batch_size=config_kitti.BATCH_SIZE,
                                          imgs_root=imgs_root_val,
                                          data_depth_velodyne_root=data_depth_velodyne_root_val,
                                          data_depth_annotated_root=data_depth_annotated_root_val,
                                          calib_velo2cam=calib_velo2cam,
                                          calib_cam2cam=calib_cam2cam)

        # save data loaders
        data_loader_train_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.KITTI_DATA_LOADER_TRAIN_FILANME)

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.KITTI_DATA_LOADER_VAL_FILENAME)

        torch.save(data_loader_train, data_loader_train_filename)
        torch.save(data_loader_val, data_loader_val_filename)


    
    # epoch_len = len(data_loader_train)
    # scheduler = PiecewiseLinear(optimizer, "lr",
    #                         milestones_values=[(65*epoch_len, 0.1), (80*epoch_len, 0.1), (85*epoch_len, 0.1), (90*epoch_len, 0.1)])
    scheduler = MultiStepLR(optimizer, milestones=[65, 80, 85, 90], gamma=0.1)
    ignite_engine = Engine(__update_model)

    # ignite_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)
    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=50), __log_training_loss)
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results)
    ignite_engine.run(data_loader_train, config_kitti.MAX_EPOCHS)
    writer.flush()
