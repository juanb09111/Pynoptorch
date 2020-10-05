# %%
"""This module performs AP evaluation using coco_eval"""

import json
import os.path
import numpy as np
import torch
import torch.nn.functional as F
from utils.tensorize_batch import tensorize_batch
from pycocotools import mask, cocoeval
from pycocotools.coco import COCO
import models
import constants
import config
import temp_variables
import sys


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()


def __results_to_json(model, data_loader_val, categories):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    res = []
    for images, anns in data_loader_val:
        images = list(img for img in images)
        images = tensorize_batch(images, device)

        model.eval()

        with torch.no_grad():
            outputs = model(images)

            for idx, out in enumerate(outputs):

                image_id = anns[idx]['image_id'].cpu().data
                pred_scores = out["scores"].cpu().data.numpy()
                pred_masks = []
                pred_boxes = []
                pred_labels = out['labels'].cpu().data.numpy()
                if "masks" in out.keys():
                    pred_masks = out["masks"].cpu().data.numpy()

                if "boxes" in out.keys():
                    pred_boxes = out["boxes"].cpu().data.numpy()

                for i, _ in enumerate(pred_scores):
                    if int(pred_labels[i]) > 0:
                        obj = {"image_id": image_id[0].numpy().tolist(),
                               "category_id": categories[int(pred_labels[i]) - 1],
                               "score": pred_scores[i].item()}
                        if "masks" in out.keys():
                            bimask = pred_masks[i] > 0.5
                            bimask = np.array(
                                bimask[0, :, :, np.newaxis], dtype=np.uint8, order="F")

                            encoded_mask = mask.encode(
                                np.asfortranarray(bimask))[0]
                            encoded_mask['counts'] = encoded_mask['counts'].decode(
                                "utf-8")
                            obj['segmentation'] = encoded_mask
                        if "boxes" in out.keys():
                            bbox = pred_boxes[i]
                            bbox_coco = [int(bbox[0]), int(bbox[1]), int(
                                bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
                            obj['bbox'] = bbox_coco

                        res.append(obj)
        torch.cuda.empty_cache()
    return res


def __export_res(model, data_loader_val, output_file, categories):
    res = __results_to_json(model, data_loader_val, categories)
    with open(output_file, 'w') as res_file:
        json.dump(res, res_file)


def mIOU(label, pred):
    # Include background
    num_classes = config.NUM_STUFF_CLASSES + config.NUM_THING_CLASSES + 1
    pred = F.softmax(pred, dim=0)
    pred = torch.argmax(pred, dim=0).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(1, num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + \
                target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


def get_mIoU(model, data_loader_val):

    iou_list = []
    for images, anns in data_loader_val:
        images = list(img for img in images)
        images = tensorize_batch(images, device)

        model.eval()
        with torch.no_grad():
            outputs = model(images)

            for idx, output in enumerate(outputs):

                label = anns[idx]["semantic_mask"]
                pred = output["semantic_logits"]
                iou = mIOU(label, pred)
                iou_list.append(iou)

    return np.mean(iou_list)


def evaluate(model=None, weights_file=None, data_loader_val=None):
    """This function performs AP evaluation using coco_eval"""

    train_res_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.RES_LOC, constants.TRAIN_RES_FILENAME)

    if weights_file is None and model is None:
        # Get model weights from config
        weights_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), config.MODEL_WEIGHTS_FILENAME)

    if model is None:
        # Get model corresponding to the one selected in config
        model = models.get_model()
        # Load model weights
        model.load_state_dict(torch.load(weights_file))
    # Set device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Empty cache
    torch.cuda.empty_cache()
    # Model to device
    model.to(device)

    if data_loader_val is None:
        # Data loader is in constants.DATA_LOADERS_LOC/constants.DATA_LOADER_VAL_FILENAME by default
        data_loader_val = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME_OBJ)

        # If DATA_LOADER is None in config then use default dataloader=data_loader_val as defined above
        data_loader_val = data_loader_val if config.DATA_LOADER is None else config.DATA_LOADER

        # Load dataloader
        data_loader_val = torch.load(data_loader_val)

    # Calculate mIoU
    average_iou = get_mIoU(model, data_loader_val)

    sys.stdout = open(train_res_file, 'a+')
    print("SemSeg mIoU = ", average_iou)
    # Annotation file is by default located under
    # constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME
    val_ann_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.COCO_ANN_LOC, constants.ANN_VAL_DEFAULT_NAME_OBJ)

    # Make coco api from annotation file
    coco_gt = COCO(val_ann_filename)

    # Get categories
    categories = list(coco_gt.cats)

    # res_filename will contain the predictions to be used later for evaluation
    res_filename = constants.COCO_RES_JSON_FILENAME
    # Export the predictions as a json file
    __export_res(model, data_loader_val, res_filename, categories)

    # Load res with coco.loadRes
    coco_dt = coco_gt.loadRes(res_filename)
    # Get the list of images
    img_ids = sorted(coco_gt.getImgIds())


    for iou_type in config.IOU_TYPES:
        sys.stdout = open(train_res_file, 'a+')
        coco_eval = cocoeval.COCOeval(coco_gt, coco_dt, iou_type)
        coco_eval.params.img_ids = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == "__main__":
    evaluate()
