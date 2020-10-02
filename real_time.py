import torch
import numpy as np
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import apply_semantic_mask_gpu, apply_panoptic_mask_gpu, apply_instance_masks
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, get_stuff_thing_classes, threshold_instances, sort_by_confidence
from utils.get_datasets import get_transform
from utils.tracker import get_tracked_objects

from PIL import Image
import time
from datetime import datetime
import os.path

result_type = None

prev_det = None
new_det = None
iou_threshold = 0.2

if config.RT_SEMANTIC:
    result_type = "semantic"

elif config.RT_INSTANCE:
    result_type = "instance"

elif config.RT_PANOPTIC:
    result_type = "panoptic"

# capture video
cap = cv2.VideoCapture(config.CAM_SOURCE)

if (cap.isOpened() == False):
    print("Unable to read camera feed")


# create timestamp
now = datetime.now()
timestamp = datetime.timestamp(now)

# Video filename
video_filename = '{}_{}_{}.avi'.format(
    config.RT_VIDEO_OUTPUT_BASENAME, result_type, timestamp)

video_full_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), config.RT_VIDEO_OUTPUT_FOLDER, video_filename)

# write video
if config.SAVE_VIDEO:
    out = cv2.VideoWriter(video_full_path, cv2.VideoWriter_fourcc(
        *'DIVX'), 6, (config.ORIGINAL_INPUT_SIZE_HW[1], config.ORIGINAL_INPUT_SIZE_HW[0]))
# empty cuda
torch.cuda.empty_cache()

# Get device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
config.DEVICE = device

# Load model
model = models.get_model()
model.load_state_dict(torch.load(config.MODEL_WEIGHTS_FILENAME))
# move model to the right device
model.to(device)

model.eval()

# model.half()

# get categories

all_categories, stuff_categories, thing_categories = get_stuff_thing_classes()

# Define transformation pipe


transforms = get_transform()


def get_seg_frame(frame, prev_det, confidence=0.5):

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)


    with torch.no_grad():

        outputs = model(images)

        threshold_preds = threshold_instances(outputs)
        sorted_preds = sort_by_confidence(threshold_preds)


        if config.OBJECT_TRACKING and result_type != "semantic":
            tracked_obj = None
            if prev_det is None:
                tracked_obj = get_tracked_objects(
                    None, sorted_preds[0]["boxes"], None, sorted_preds[0]["labels"], iou_threshold)
            else:
                tracked_obj = get_tracked_objects(
                    prev_det[0]["boxes"], sorted_preds[0]["boxes"], prev_det[0]["labels"], sorted_preds[0]["labels"], iou_threshold)


            sorted_preds[0]["ids"] = tracked_obj

            if len(tracked_obj) > 0:
                sorted_preds[0]["boxes"] = sorted_preds[0]["boxes"][:len(
                    tracked_obj)]
                sorted_preds[0]["masks"] = sorted_preds[0]["masks"][:len(
                    tracked_obj)]
                sorted_preds[0]["scores"] = sorted_preds[0]["scores"][:len(
                    tracked_obj)]
                sorted_preds[0]["labels"] = sorted_preds[0]["labels"][:len(
                    tracked_obj)]


        if result_type == "semantic":

            logits = outputs[0]["semantic_logits"]
            mask = torch.argmax(logits, dim=0)
            im = apply_semantic_mask_gpu(images[0], mask, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES)

            return im.cpu().permute(1, 2, 0).numpy(), None, None

        if result_type == "instance":
            
            im = apply_instance_masks(images[0], sorted_preds[0]['masks'], 0.5, ids=sorted_preds[0]["ids"])

            return im.cpu().permute(1, 2, 0).numpy(), None, sorted_preds

        if result_type == "panoptic" and len(sorted_preds[0]["masks"]) > 0:

            # Get intermediate prediction and semantice prediction
            inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                sorted_preds, all_categories, stuff_categories, thing_categories, threshold_by_confidence=False, sort_confidence=False)
            canvas = panoptic_canvas(
                inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]

            if canvas is None:
                return frame, summary_batch, sorted_preds
            else:

                im = apply_panoptic_mask_gpu(
                    images[0], canvas).cpu().permute(1, 2, 0).numpy()

                return im, summary_batch, sorted_preds

        else:
            return frame, None, sorted_preds


# Video loop
while(True):

    ret, frame = cap.read()
    start = time.time_ns()
    
    im, summary_batch, new_det = get_seg_frame(frame, prev_det, confidence=0.5)
    prev_det = new_det
    end = time.time_ns()
    fps = round(1/((end-start)/1e9), 1)


    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale = 0.4
    fontColor = (255, 255, 255)
    lineType = 1
    text = "fps: {} - ".format(fps)
    if summary_batch:
        summary = summary_batch[0]
        for obj in summary:
            text = text + \
                '{}: {}'.format(obj["name"], obj["count_obj"]) + " - "

    im = cv2.UMat(im)

    cv2.putText(im, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    if config.SAVE_VIDEO:
        out.write(np.uint8(im.get().astype('f')*255))

    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if config.SAVE_VIDEO:
    out.release()

cv2.destroyAllWindows()
