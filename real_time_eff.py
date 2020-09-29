import torch
import numpy as np
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import colors_pallete, apply_semantic_mask_gpu, apply_mask, apply_panoptic_mask_gpu, randRGB
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, get_stuff_thing_classes, threshold_instances, sort_by_confidence
from utils.get_datasets import get_transform
from utils.tracker_2 import get_tracked_objects
import matplotlib.pyplot as plt
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
# cap = cv2.VideoCapture(config.CAM_DEVICE)
cap = cv2.VideoCapture("plain.avi")

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

    # start_real = time.time_ns()
    # start_image = time.time_ns()
    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)
    # stop_image = time.time_ns()
    # time_image = stop_image - start_image

    with torch.no_grad():

        # start_model = time.time_ns()
        outputs = model(images)
        # stop_model = time.time_ns()
        # time_model = stop_model - start_model
        # print("model", 1/((stop_model - start_model)/1e9))

        # sort_start = time.time_ns()
        threshold_preds = threshold_instances(outputs)
        sorted_preds = sort_by_confidence(threshold_preds)
        # sort_stop = time.time_ns()
        # time_sort = sort_stop - sort_start

        # start_trk = time.time_ns()
        tracked_obj = None
        if prev_det is None:
            tracked_obj = get_tracked_objects(
                None, sorted_preds[0]["boxes"], None, sorted_preds[0]["labels"], iou_threshold)
        else:
            tracked_obj = get_tracked_objects(
                prev_det[0]["boxes"], sorted_preds[0]["boxes"], prev_det[0]["labels"], sorted_preds[0]["labels"], iou_threshold)
        # stop_trk = time.time_ns()
        # trk_time = stop_trk - start_trk
        # print("trk", 1/((stop_trk - start_trk)/1e9))

        #

        # filter_start = time.time_ns()
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

        # filter_stop = time.time_ns()
        # filter_time = filter_stop - filter_start

        if result_type == "semantic":
            # sem_start = time.time_ns()
            logits = outputs[0]["semantic_logits"]
            mask = torch.argmax(logits, dim=0)
            im = apply_semantic_mask_gpu(images[0], mask, colors_pallete)
            # sem_end = time.time_ns()
            # print("semantic seg fps: ", 1/((sem_end-sem_start)/1e9))
            return im.cpu().permute(1, 2, 0).numpy(), None

        if result_type == "instance":
            # ins_start = time.time_ns()
            masks = outputs[0]['masks']
            scores = outputs[0]['scores']
            labels = outputs[0]['labels']
            im = frame/255
            for i, mask in enumerate(masks):
                if scores[i] > confidence and labels[i] > 0:
                    mask = mask[0]
                    mask = mask.cpu().numpy()
                    mask_color = randRGB()
                    im = apply_mask(im, mask, mask_color, confidence)
            # ins_end = time.time_ns()
            # print("instance seg fps: ", 1/((ins_end-ins_start)/1e9))
            return im, None

        if result_type == "panoptic" and len(sorted_preds[0]["masks"]) > 0:
            # start_pan = time.time_ns()
            # Get intermediate prediction and semantice prediction
            inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                sorted_preds, all_categories, stuff_categories, thing_categories, threshold_by_confidence=False, sort_confidence=False)
            canvas = panoptic_canvas(
                inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]

            # print("trans", 1/((trans_stop - trans_start)/1e9))
            # stop_pan = time.time_ns()
            # pan_time = stop_pan - start_pan
            # print("pan", 1/((stop_pan - start_pan)/1e9))
            if canvas is None:
                # trans_stop = time.time_ns()
                # print("trans", 1/((trans_stop - trans_start)/1e9))
                return frame, summary_batch, sorted_preds
            else:
                # start_mask = time.time_ns()
                im = apply_panoptic_mask_gpu(
                    images[0], canvas).cpu().permute(1, 2, 0).numpy()
                # stop_mask = time.time_ns()
                # mask_time = stop_mask - start_mask

                # total_time = time_image + time_model + time_sort + trk_time + filter_time + pan_time + mask_time
                # print("trk_time", trk_time)
                # print(time_image*100/total_time, time_model*100/total_time, time_sort*100/total_time, trk_time * 100/total_time, filter_time*100/total_time, pan_time*100/total_time, mask_time*100/total_time)
                # stop_real = time.time_ns()
                # time_real = stop_real- start_real
                # print((total_time-time_real)/1e9, total_time)
                # print("mask", 1/((stop_mask - start_mask)/1e9))
                # trans_stop = time.time_ns()

                return im, summary_batch, sorted_preds

        else:
            # print("trans", 1/((trans_stop - trans_start)/1e9))
            return frame, None, sorted_preds


# Video loop
while(True):

    ret, frame = cap.read()
    start = time.time_ns()
    frame = cv2.resize(frame, (640, 480))
    im, summary_batch, new_det = get_seg_frame(frame, prev_det, confidence=0.5)
    prev_det = new_det
    end = time.time_ns()
    fps = round(1/((end-start)/1e9), 1)
    # print("fps", end-start)

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
