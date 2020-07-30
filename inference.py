import config
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch

from datetime import datetime


from utils import show_bbox, panoptic_fusion, get_datasets
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

    # Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    # load weights
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
                panoptic_fusion.get_panoptic_results(
                    images, outputs, folder, j)
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

    test_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), constants.TEST_DIR)
    data_loader_test = get_datasets.get_dataloaders(
        config.BATCH_SIZE, test_dir, is_test_set=True)

    confidence = 0.5
    model = models.get_model()

    if config.INSTANCE:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "instance",
                   '{}_{}_results_instance_{}'.format(config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.SEMANTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "semantic",
                   '{}_{}_results_semantic_{}'.format(config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.PANOPTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "panoptic",
                   '{}_{}_results_panoptic_{}'.format(config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)
