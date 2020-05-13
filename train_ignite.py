# %%
import os.path
from ignite.engine import Events, Engine
import torch
from utils import get_datasets
from evaluate import mean_ap

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
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()}
                   for t in annotations]
    loss_dict = model(imgs, annotations)
    losses = sum(loss for loss in loss_dict.values())
    losses.backward()
    optimizer.step()
    return losses



def __eval_model():
    img_results = {}
    for images, gt_dict in data_loader_val:
        images = list(img.to(device) for img in images)
        gt_dict = [{k: v.to(device) for k, v in t.items()}
                   for t in gt_dict]
        model.eval()
        with torch.no_grad():
            pred_dict = model(images)
            img_id = gt_dict[0]["image_id"].item()
            img_results["img_id_{}".format(img_id)] = mean_ap.get_single_image_results(
                gt_dict[0], pred_dict[0], 0.5)

    precision, recall = mean_ap.calc_precision_recall(img_results)
    return precision, recall

# %% Define Event listeners

def __log_training_loss(trainer_engine):
    batch_loss = trainer_engine.state.output
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    i = trainer_engine.state.iteration
    print("Epoch {}/{} : {} - batch loss: {:.2f}".format(state_epoch, max_epochs, i, batch_loss))


def __log_validation_results(trainer_engine):
    state_epoch = trainer_engine.state.epoch
    max_epochs = trainer_engine.state.max_epochs
    precision, recall = __eval_model()
    weights_path = "{}{}_precision_{}_recall_{}.pth".format(
        constants.MODELS_LOC, config.MODEL_WEIGHTS_FILENAME_PREFIX, precision, recall)
    state_dict = model.state_dict()
    torch.save(state_dict, weights_path)
    print(
        "Validation Results - Epoch {}/{} precision: {:.2f}, recall: {:.2f}"
        .format(state_epoch, max_epochs, precision, recall))


if __name__ == "__main__":

    # Get model according to config
    model = models.get_model()
    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    # Empty cuda cache
    torch.cuda.empty_cache()
    # move model to the right device
    model.to(device)

    # Define params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    data_loader_train = None
    data_loader_val = None
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

        train_ann_filename = train_ann_filename if config.COCO_ANN_TRAIN is None else config.COCO_ANN_TRAIN
        val_ann_filename = val_ann_filename if config.COCO_ANN_VAL is None else config.COCO_ANN_VAL
        # overwrite config
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

        data_loader_train = get_datasets.get_dataloaders(
            config.BATCH_SIZE, train_dir, coco_ann_train)

        data_loader_val = get_datasets.get_dataloaders(
            config.BATCH_SIZE, val_dir, coco_ann_val)

        # save data loaders
        data_loader_train_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_TRAIN_FILANME)

        data_loader_val_filename = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME)

        torch.save(data_loader_train, data_loader_train_filename)
        torch.save(data_loader_val, data_loader_val_filename)

    ignite_engine = Engine(__update_model)
    ignite_engine.add_event_handler(Events.ITERATION_COMPLETED(every=50), __log_training_loss)
    ignite_engine.add_event_handler(Events.EPOCH_COMPLETED, __log_validation_results)
    ignite_engine.run(data_loader_train, max_epochs=25)
