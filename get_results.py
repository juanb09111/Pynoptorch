import config
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch


from utils import show_bbox, panoptic_fusion
import glob


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

config.DEVICE = device
torch.cuda.empty_cache()


def view_masks(model,
               data_loader_val,
               num_classes,
               weights_file,
               result_type,
               folder,
               confidence=0.5):

    #Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)
    i = 0
    j = 0
    for images, _ in data_loader_val:
        images = list(img for img in images)
        images = tensorize_batch(images, device)

        model.eval()
        with torch.no_grad():
            outputs = model(images)

            if result_type == "panoptic":
                panoptic_fusion.get_panoptic_results(images, outputs, folder, j)
                j += 1
                torch.cuda.empty_cache()

            for idx, output in enumerate(outputs):
                i += 1
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                if result_type == "instance":
                    show_bbox.overlay_masks(
                        img, output, confidence, folder, i)
                elif result_type == "semantic":
                    show_bbox.get_semantic_masks(
                        img, output, num_classes, folder, i)

                torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()

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

    view_masks(model, data_loader_val, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
               config.MODEL_WEIGHTS_FILENAME,
               "semantic",
               '{}_results_semantic_resnet'.format(config.MODEL),
               confidence=0.5)
    
    view_masks(model, data_loader_val, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
               config.MODEL_WEIGHTS_FILENAME,
               "panoptic",
               '{}_results_panoptic_resnet'.format(config.MODEL),
               confidence=0.5)
