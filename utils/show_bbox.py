# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import random
import os
import os.path
import json
import numpy as np
import numpy.ma as ma
import torch
from PIL import Image
from os import listdir
from os.path import isfile, join
import re
import glob


def randRGB():
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb


colors_pallete = [
    [96/255, 99/255, 225/255],  # blue
    [96/255, 99/255, 225/255],  # blue
    [191/255, 96/255, 225/255],  # purple
    [245/255, 23/255, 23/255],  # red
    [69/255, 249/255, 60/255],  # green
    [253/255, 170/255, 3/255],  # orange
    [3/255, 218/255, 253/255],  # sky blue
    [236/255, 253/255, 3/255],  # yellow
]


def get_colors_palete(num_classes):
    colors = [randRGB() for i in range(num_classes)]
    return colors_pallete


def show_masks(img, preds, confidence, colors, file_name):
    masks = preds['masks']
    scores = preds['scores']
    labels = preds['labels']
    my_path = os.path.dirname(__file__)
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    for i, mask in enumerate(masks):
        if scores[i] > confidence and labels[i] > 0:
            mask = mask[0]
            mask = mask.numpy()
            image = np.zeros((mask.shape[0], mask.shape[1], 4))
            # set transparency to 0
            image[:, :, 3] = 0
            locs = np.where(mask > confidence, 1, 0)
            locs = np.transpose(np.nonzero(locs))
            mask_color = colors[labels[i]]

            for loc in locs:
                image[loc[0], loc[1], :] = [*mask_color, 0.5]
            ax.imshow(image)
    plt.axis('off')
    # plt.show()

    fig.savefig(os.path.join(
        my_path, '../maskRCNN_results_mask/{}.png'.format(file_name)))


def apply_mask(image, mask, color, confidence, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > confidence,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


def overlay_masks(img, preds, confidence, colors, folder, file_name):
    masks = preds['masks']
    scores = preds['scores']
    labels = preds['labels']
    my_path = os.path.dirname(__file__)
    width = img.shape[1]
    height = img.shape[0]
    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for i, mask in enumerate(masks):
        if scores[i] > confidence and labels[i] > 0:
            mask = mask[0]
            mask = mask.cpu().numpy()
            mask_color = colors[labels[i]]
            im = apply_mask(img, mask, mask_color, confidence)
            ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join(
        my_path, '../{}/{}.png'.format(folder, file_name)))
    plt.close(fig)


def get_semantic_masks(img, preds, folder, file_name):
    logits = preds["semantic_logits"]
    mask = torch.argmax(logits, dim=0)
    mask = mask.cpu().numpy()
    my_path = os.path.dirname(__file__)
    width = img.shape[1]
    height = img.shape[0]
    dppi = 96
    fig, ax = plt.subplots(1, 1, figsize=(width/dppi, height/dppi), dpi=dppi)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    
    plt.axis('off')
    ax.imshow(img)
    ax.imshow(mask, cmap='jet', alpha=0.3)
    fig.savefig(os.path.join(
        my_path, '../{}/{}.png'.format(folder, file_name)))
    plt.close(fig)
