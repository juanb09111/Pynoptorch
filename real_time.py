import torch
import numpy as np
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import colors_pallete, apply_semantic_mask_gpu, apply_mask, apply_panoptic_mask_gpu, randRGB
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, get_stuff_thing_classes
from utils.get_datasets import get_transform
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime
import os.path

result_type = None

if config.RT_SEMANTIC:
    result_type = "semantic"

elif config.RT_INSTANCE:
    result_type = "instance"

elif config.RT_PANOPTIC:
    result_type = "panoptic"

# capture video
cap = cv2.VideoCapture(config.CAM_DEVICE)
# cap = cv2.VideoCapture("plain.avi")

if (cap.isOpened() == False):
    print("Unable to read camera feed")


# create timestamp
now = datetime.now()
timestamp = datetime.timestamp(now)

# Video filename
video_filename = '{}_{}_{}.avi'.format(config.RT_VIDEO_OUTPUT_BASENAME, result_type, timestamp)

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


def get_seg_frame(frame, confidence=0.5):

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)

    with torch.no_grad():
        # start = time.time_ns()
        outputs = model(images)
        
        # print("model in-out fps: ", 1/((end-start)/1e9))

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

        if result_type == "panoptic" and len(outputs[0]['masks']) > 0:
            # pan_start = time.time_ns()
            inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                outputs, all_categories, stuff_categories, thing_categories)
            canvas = panoptic_canvas(
                inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]
            if canvas is None:
                return frame, summary_batch
            else:
                im = apply_panoptic_mask_gpu(images[0], canvas)
                # pan_end = time.time_ns()
                # print("panoptic fps: ", 1/((pan_end-pan_start)/1e9))
                # end = time.time_ns()
                # fps = 1/((end-start)/1e9)
                return im.cpu().permute(1, 2, 0).numpy(), summary_batch

        else:
            return frame, None



# Video loop
while(True):
    start = time.time_ns()

    ret, frame = cap.read()
    im, summary_batch = get_seg_frame(frame, confidence=0.5)

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
            text = text + '{}: {}'.format(obj["name"], obj["count_obj"]) + " - "

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
