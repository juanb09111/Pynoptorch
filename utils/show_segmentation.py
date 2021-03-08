# %%
import matplotlib.pyplot as plt
import random
import os
import os.path
import numpy as np
import torch
from os.path import join
import config
import temp_variables
import time
import torch.nn.functional as F
import cv2



def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb




def get_colors_palete(num_classes):
    if (config.SEM_COLORS != None) and (len(config.SEM_COLORS) >= num_classes + 1):

        return config.SEM_COLORS
    
    colors = [randRGB(i+5) for i in range(num_classes + 1)]
    return colors

def apply_instance_masks(image, masks, confidence, ids=None):

    masks = masks.squeeze(1)
    
    background_mask = torch.zeros((1, *masks[0].shape), device=temp_variables.DEVICE)
    background_mask = background_mask.new_full(background_mask.shape, confidence)
    
    masks = torch.cat([background_mask, masks], dim=0)
    
    mask_argmax = torch.argmax(masks, dim=0)
    
    if ids is not None:
        mask = torch.tensor(background_mask, dtype=torch.long, device=temp_variables.DEVICE)
        for idx, obj_id in enumerate(ids):
            mask = torch.where((mask_argmax == idx + 1), obj_id, mask)
    else:
        mask = mask_argmax

    max_val = mask.max()

    for i in range(1, max_val + 1):
        for c in range(3):
            alpha = 0.45
            color = randRGB(i)
            image[c, :, :] = torch.where(mask == i,
                                      image[c, :, :] *
                                      (1 - alpha) + alpha * color[c],
                                      image[c, :, :])
    return image


def apply_semantic_mask_gpu(image, mask, num_classes):
    colors = get_colors_palete(num_classes)
    # print("---------------")
    # print(colors)
    # print("---------------")
    max_val = mask.max()
    for i in range(1, max_val + 1):
        for c in range(3):
            
            alpha = 0.45
            color = colors[i]
            image[c, :, :] = torch.where(mask == i,
                                      image[c, :, :] *
                                      (1 - alpha) + alpha * color[c],
                                      image[c, :, :])
    
    return image

def apply_panoptic_mask_gpu(image, mask):

    num_stuff_classes = config.NUM_STUFF_CLASSES
    max_val = mask.max()
    colors = get_colors_palete(config.NUM_STUFF_CLASSES + config.NUM_THING_CLASSES)

    for i in range(1, max_val + 1):
        for c in range(3):

            if i <= num_stuff_classes:
                color = colors[i]
                alpha = 0.45

            else:
                alpha = 0.45
                color = randRGB(i)
            
            image[c, :, :] = torch.where(mask == i,
                                      image[c, :, :] *
                                      (1 - alpha) + alpha * color[c],
                                      image[c, :, :])
    return image



def draw_bboxes(im, boxes, thing_categories, color_max_val=1, labels=None, ids=None):
    im = cv2.UMat(im)
    for i, box in enumerate(boxes):

        top_left = (box[0], box[1])
        bottom_right = (box[2], box[3])
        cv2.rectangle(im,top_left, bottom_right, (0,0,0),6)

        if labels != None:
            label = labels[i].item()
            label_name = thing_categories[label - 1]["name"]
            labelSize = cv2.getTextSize(str(label_name), cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            _x1 = box[0]
            _y1 = box[1]
            _x2 = _x1 + int(labelSize[0][0])
            _y2 = _y1 + int(labelSize[0][1])
            cv2.rectangle(im,(_x1,_y1),(_x2,_y2),(0,color_max_val,0),cv2.FILLED)
            cv2.putText(im, str(label_name), (_x1, _y2), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
        
        if ids != None:
            obj_id = ids[i].item()
            text = "id={}".format(obj_id)
            labelSize = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
            _x1 = box[0]
            _y1 = box[3]
            _x2 = _x1 + int(labelSize[0][0])
            _y2 = _y1 - int(labelSize[0][1])
            cv2.rectangle(im,(_x1,_y1),(_x2,_y2),(0,color_max_val,0),cv2.FILLED)
            cv2.putText(im, str(text), (_x1, _y1), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1)
    return im

def save_fig(im, folder, file_name, tensor_image = True, cv_umat=False):

    if tensor_image:
        im = im.cpu().permute(1, 2, 0).numpy()
    
    if cv_umat:
        im = cv2.UMat.get(im)
    file_name_basename = os.path.basename(file_name)
    filename = os.path.splitext(file_name_basename)[0]

    my_path = os.path.dirname(__file__)
    width = im.shape[1]
    height = im.shape[0]
    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join(
        my_path, '../{}/{}.png'.format(folder, filename)))
    plt.close(fig)
    