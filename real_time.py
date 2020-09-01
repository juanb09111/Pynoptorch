import torch
import numpy as np
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import colors_pallete, apply_semantic_mask, apply_mask, apply_panoptic_mask_gpu, randRGB
from utils.panoptic_fusion import panoptic_fusion, panoptic_canvas, get_stuff_thing_classes
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

#capture video
cap = cv2.VideoCapture(config.CAM_DEVICE)
# cap = cv2.VideoCapture("plain.avi")

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
def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(custom_transforms)


transforms = get_transform()


def get_seg_frame(frame, confidence=0.5):

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)

    with torch.no_grad():
        start = time.time_ns()
        outputs = model(images)
        end = time.time_ns()
        print("model in-out fps: ", 1/((end-start)/1e9))

        if result_type == "semantic":
            sem_start = time.time_ns()
            logits = outputs[0]["semantic_logits"]
            mask = torch.argmax(logits, dim=0)
            mask = mask.cpu().numpy()
            im = apply_semantic_mask(frame/255, mask, colors_pallete)
            sem_end = time.time_ns()
            print("semantic seg fps: ", 1/((sem_end-sem_start)/1e9))
            return im
        
        if result_type == "instance":
            ins_start = time.time_ns()
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
            ins_end = time.time_ns()
            print("instance seg fps: ", 1/((ins_end-ins_start)/1e9))
            return im
        
        if result_type == "panoptic" and len(outputs[0]['masks']) > 0:
            pan_start = time.time_ns()
            inter_pred_batch, sem_pred_batch = panoptic_fusion(outputs, all_categories, stuff_categories, thing_categories)
            canvas = panoptic_canvas(inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]
            if canvas is None:
                return frame
            else:
                im = apply_panoptic_mask_gpu(images[0], canvas)
                pan_end = time.time_ns()
                print("panoptic fps: ", 1/((pan_end-pan_start)/1e9))
                return im.cpu().permute(1, 2, 0).numpy()
        
        else:
            return frame




# Video loop
while(True):
    ret, frame = cap.read()
    im = get_seg_frame(frame, confidence=0.5)
    
    if config.SAVE_VIDEO:
        out.write(np.uint8(im*255))

    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if config.SAVE_VIDEO:
    out.release()

cv2.destroyAllWindows()
