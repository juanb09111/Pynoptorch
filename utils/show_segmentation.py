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


sem_colors = [
    [0/255, 0/255, 0/255],
    [30/255, 95/255, 170/255],
    [245/255, 208/255, 58/255],
    [224/255, 67/255, 217/255],
    [255/255, 255/255, 153/255],
    [177/255, 89/255, 40/255],
    [166/255, 3/255, 3/255],
    [65/255, 117/255, 5/255],
    [94/255, 245/255, 184/255],
    [140/255, 190/255, 248/255]
]

def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb


# def get_colors_palete(num_classes):
#     colors = [randRGB(i+5) for i in range(num_classes + 1)]
#     return colors

def get_colors_palete(num_classes):
    # colors = [randRGB(i+5) for i in range(num_classes + 1)]
    # return colors
    return sem_colors

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
    colors = get_colors_palete(config.NUM_THING_CLASSES + config.NUM_THING_CLASSES)

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


def save_fig(im, folder, file_name):

    im = im.cpu().permute(1, 2, 0).numpy()
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
    