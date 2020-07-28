# %%
import os.path
from ignite.engine import Events, Engine
import torch
from utils import get_datasets
from utils.tensorize_batch import tensorize_batch
from eval_coco import evaluate

import config
import constants
import models
import map_hasty
import get_splits

# %%


def __update_model(_, batch):
    model.train()
    optimizer.zero_grad()
    imgs, annotations = batch[0], batch[1]
    imgs = list(img for img in imgs)
    # imgs = tensorize_batch(imgs, device)
    # annotations = [{k: v.to(device) for k, v in t.items()}
    #                for t in annotations]

    # Array idexes of anns with no instance annotations
    no_instance_anns_idx = []
    # Array idexes of anns with instance annotations
    instance_anns_idx = []
    for idx, ann in enumerate(annotations):
        if ann["num_instances"] == 0:
            no_instance_anns_idx.append(idx)
        else:
            instance_anns_idx.append(idx)

    losses = 0
    if len(no_instance_anns_idx) > 0:
        # For anns without instance anns deactivate instance branch
        anns = [annotations[idx] for idx in no_instance_anns_idx]
        no_instance_imgs = [imgs[idx] for idx in no_instance_anns_idx]

        no_instance_imgs = tensorize_batch(no_instance_imgs, device)
        anns = [{k: v.to(device) for k, v in t.items()}
                for t in anns]

        loss_dict = model(no_instance_imgs, anns=anns, instance=False)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        if len(no_instance_anns_idx) < len(annotations):
            # For the rest of anns with instance anns activate instance branch
            anns = [annotations[idx] for idx in instance_anns_idx]
            instance_imgs = [imgs[idx] for idx in instance_anns_idx]

            instance_imgs = tensorize_batch(no_instance_imgs, device)
            anns = [{k: v.to(device) for k, v in t.items()}
                    for t in anns]

            loss_dict = model(instance_imgs, anns=anns)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
    else:
        # If no "no instance anns" then run model with both semantic and instance branch active
        imgs = tensorize_batch(imgs, device)
        annotations = [{k: v.to(device) for k, v in t.items()}
                       for t in annotations]
        loss_dict = model(imgs, anns=annotations)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

    return losses


# %% Define Event listeners


def __log_training_loss(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    i = trainer_engine.state.iteration
    print("Epoch {}/{} : {} - batch loss: {:.2f}".format(state_epoch,
                                                         max_epochs, i, batch_loss))


def __log_validation_results(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    weights_path = "{}{}_{}_loss_{}.pth".format(
        constants.MODELS_LOC, config.MODEL_WEIGHTS_FILENAME_PREFIX, config.BACKBONE, batch_loss)
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)
    print("Validation Results - Epoch {}/{} batch_loss: {:.2f}".format(state_epoch,
                                                                       max_epochs, batch_loss))
    evaluate(model=model, weights_file=weights_path,
             data_loader_val=data_loader_val_obj)
    torch.cuda.empty_cache()


if __name__ == "__main__":

    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    config.DEVICE = device
    # Empty cuda cache
    torch.cuda.empty_cache()

    # Get model according to config
    model = models.get_model()
    # move model to the right device
    model.to(device)

    print(torch.cuda.memory_allocated(device=device))
    # Define params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    data_loader_train = None
    data_loader_val = None
    data_loader_val_obj = None

    if config.USE_PREEXISTING_DATA_LOADERS:
        data_loader_train = torch.load(config.DATA_LOADER_TRAIN_FILANME)
        data_loader_val = torch.load(config.DATA_LOADER_VAL_FILENAME)

    else:
        if config.AUTOMATICALLY_SPLIT_SETS:
            get_splits.get_splits()

        # get annotations file names
        train_ann_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_TRAIN_DEFAULT_NAME)

        val_ann_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME)

        val_ann_obj_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME_OBJ)

        train_ann_filename = train_ann_filename if config.COCO_ANN_TRAIN is None else config.COCO_ANN_TRAIN
        val_ann_filename = val_ann_filename if config.COCO_ANN_VAL is None else config.COCO_ANN_VAL
        # overwrite config

        # TODO: Next two lines may not be necesary
        config.COCO_ANN_VAL = val_ann_filename
        config.COCO_ANN_TRAIN = train_ann_filename
        # write annotations json files for every split
        map_hasty.get_split(constants.TRAIN_DIR, train_ann_filename)
        map_hasty.get_split(constants.VAL_DIR, val_ann_filename)

        train_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.TRAIN_DIR)

        val_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.VAL_DIR)

        coco_ann_train = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), train_ann_filename)

        coco_ann_val = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), val_ann_filename)

        coco_ann_val_obj = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), val_ann_obj_filename)

        # data loaders
        data_loader_train = get_datasets.get_dataloaders(
            config.BATCH_SIZE, train_dir, coco_ann_train, semantic_masks_folder=config.SEMANTIC_SEGMENTATION_DATA)

        data_loader_val = get_datasets.get_dataloaders(
            config.BATCH_SIZE, val_dir, coco_ann_val, semantic_masks_folder=config.SEMANTIC_SEGMENTATION_DATA)

        data_loader_val_obj = get_datasets.get_dataloaders(
            config.BATCH_SIZE, val_dir, coco_ann_val_obj, semantic_masks_folder=config.SEMANTIC_SEGMENTATION_DATA)

        # save data loaders
        data_loader_train_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_TRAIN_FILANME)

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME)
        
        data_loader_val_obj_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME_OBJ)

        torch.save(data_loader_train, data_loader_train_filename)
        torch.save(data_loader_val, data_loader_val_filename)
        torch.save(data_loader_val_obj, data_loader_val_obj_filename)

    ignite_engine = Engine(__update_model)
    ignite_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=50), __log_training_loss)
    ignite_engine.add_event_handler(
        Events.EPOCH_COMPLETED, __log_validation_results)
    ignite_engine.run(data_loader_train, config.MAX_EPOCHS)
