import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image

import config
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
import numpy as np
import torch
import json

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.patches as patches
import cv2
import random
from utils import show_bbox, fasterrcnn_resnet50_fpn, get_datasets, iss_model
import glob
import re


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
torch.cuda.empty_cache()
# move model to the right device


def view_masks(model,
               data_loader_val,
               num_classes,
               weights_file,
               result_type,
               folder,
               confidence=0.5):
    model.load_state_dict(torch.load(weights_file))
    model.to(device)
    colors = show_bbox.get_colors_palete(num_classes)
    i = 0
    for images, _ in data_loader_val:
        images = list(img for img in images)
        images = tensorize_batch(images, device)

        model.eval()
        with torch.no_grad():
            outputs = model(images)

            for idx, output in enumerate(outputs):
                i += 1
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                if result_type == "instance":
                    show_bbox.overlay_masks(
                        img, output, confidence, colors, folder, i)
                elif result_type == "semantic":
                    show_bbox.get_semantic_masks(
                        img, output, folder, i)

                torch.cuda.empty_cache()


if __name__ == "__main__":
    data_loader_val_filename = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.DATA_LOADERS_LOC, constants.DATA_LOADER_VAL_FILENAME)
    data_loader_val = torch.load(data_loader_val_filename)

    confidence = 0.5
    model = models.get_model()
    view_masks(model, data_loader_val, config.NUM_THING_CLASSES,
               config.MODEL_WEIGHTS_FILENAME,
               "instance",
               '{}_results_mask_resnet'.format(config.MODEL),
               confidence=0.5)

    view_masks(model, data_loader_val, config.NUM_THING_CLASSES,
               config.MODEL_WEIGHTS_FILENAME,
               "semantic",
               '{}_results_semantic_resnet'.format(config.MODEL),
               confidence=0.5)
