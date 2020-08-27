import torch
import numpy as np
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import colors_pallete, apply_semantic_mask
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import datetime
import os.path

#capture video
cap = cv2.VideoCapture(config.CAM_DEVICE)

if (cap.isOpened() == False): 
    print("Unable to read camera feed")


#create timestamp
now = datetime.now()
timestamp = datetime.timestamp(now)

# Video filename
video_filename = '{}_{}.avi'.format(config.RT_VIDEO_OUTPUT_BASENAME, timestamp)

video_full_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), config.RT_VIDEO_OUTPUT_FOLDER, video_filename)

#write video
if config.SAVE_VIDEO:
    out = cv2.VideoWriter(video_full_path, cv2.VideoWriter_fourcc(
        *'DIVX'), 6, (config.ORIGINAL_INPUT_SIZE_HW[1], config.ORIGINAL_INPUT_SIZE_HW[0]))
#empty cuda
torch.cuda.empty_cache()

# Get device
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
model = models.get_model()
model.load_state_dict(torch.load(config.MODEL_WEIGHTS_FILENAME))
# move model to the right device
model.to(device)

model.eval()

# model.half()

# Define transformation pipe
def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(custom_transforms)


transforms = get_transform()

# Render masks
def view_masks(frame,
               num_classes,
               result_type,
               confidence=0.5):

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)
    output = model(images)[0]
    if result_type == "semantic":

        logits = output["semantic_logits"]
        mask = torch.argmax(logits, dim=0)
        mask = mask.cpu().numpy()
        im = apply_semantic_mask(frame/255, mask, colors_pallete)
        return im

# Video loop
while(True):
    ret, frame = cap.read()

    if config.RT_SEMANTIC:
        # start = time.time_ns()
        im = view_masks(frame, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                        "semantic",
                        confidence=0.5)
        if config.SAVE_VIDEO:
            out.write(np.uint8(im*255))
        # end = time.time_ns()
        # print("time: ", end - start)
        cv2.imshow('frame', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if config.SAVE_VIDEO:
    out.release()

cv2.destroyAllWindows()
