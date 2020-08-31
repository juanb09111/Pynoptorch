# %%
import matplotlib.pyplot as plt
import random
import os
import os.path
import numpy as np
import torch
from os.path import join
import config
import time


def randRGB(seed=None):
    if seed is not None:
        random.seed(seed)
    r = random.random()
    g = random.random()
    b = random.random()
    rgb = [r, g, b]
    return rgb


colors_pallete = [
    [255/255, 255/255, 255/255],  # white
    [96/255, 99/255, 225/255],  # blue
    [191/255, 96/255, 225/255],  # purple
    [245/255, 23/255, 23/255],  # red
    [69/255, 249/255, 60/255],  # green
    [253/255, 170/255, 3/255],  # orange
    [3/255, 218/255, 253/255],  # sky blue
    [236/255, 253/255, 3/255]  # yellow
]


def get_colors_palete(num_classes):
    colors = [randRGB(i+15) for i in range(num_classes + 1)]
    return colors


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


def apply_semantic_mask(image, mask, colors):
    
    max_val = mask.max()
    for i in range(0, max_val + 1):
        for c in range(3):
            if i == 0:
                alpha = 0.25
            else:
                alpha = 0.45
            color = colors[i]
            image[:, :, c] = np.where(mask == i,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
    
    return image

def apply_panoptic_mask(image, mask):

    num_stuff_classes = config.NUM_STUFF_CLASSES
    max_val = mask.max()
    colors = get_colors_palete(max_val)

    for i in range(0, max_val + 1):
        for c in range(3):
            if i == 0:
                alpha = 0.25
                color = colors[i]

            elif i <= num_stuff_classes:
                color = colors[i]
                alpha = 0.45

            else:
                alpha = 0.45
                color = randRGB(i+10)
            
            image[:, :, c] = np.where(mask == i,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
    return image

def overlay_masks(img, preds, confidence, folder, file_name):

    start = time.time_ns()

    masks = preds['masks']
    scores = preds['scores']
    labels = preds['labels']

    # file_name_basename = os.path.basename(file_name)
    # filename = os.path.splitext(file_name_basename)[0]
    
    # width = img.shape[1]
    # height = img.shape[0]
    # dppi = 96
    # fig, ax = plt.subplots(1, 1, figsize=(width/dppi, height/dppi), dpi=dppi)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                     hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    for i, mask in enumerate(masks):
        if scores[i] > confidence and labels[i] > 0:
            mask = mask[0]
            mask = mask.cpu().numpy()
            mask_color = randRGB()
            im = apply_mask(img, mask, mask_color, confidence)
    end = time.time_ns()
    print("instance seg fps: ", 1/((end-start)/1e9))
    #         ax.imshow(im,  interpolation='nearest', aspect='auto')
    # plt.axis('off')

    # my_path = os.path.dirname(__file__)
    # fig.savefig(os.path.join(
    #     my_path, '../{}/{}.png'.format(folder, filename)))
    # plt.close(fig)


def get_semantic_masks(img, preds, num_classes, folder, file_name):

    start = time.time_ns()
    
    logits = preds["semantic_logits"]
    mask = torch.argmax(logits, dim=0)
    mask = mask.cpu().numpy()
    
    colors = colors_pallete
    im = apply_semantic_mask(img, mask, colors)

    end = time.time_ns()
    print("semantic seg fps: ", 1/((end-start)/1e9))

    file_name_basename = os.path.basename(file_name)
    filename = os.path.splitext(file_name_basename)[0]

    # colors = get_colors_palete(num_classes)

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

    ax.imshow(im,  interpolation='nearest', aspect='auto')
    plt.axis('off')
    fig.savefig(os.path.join(
        my_path, '../{}/{}.png'.format(folder, filename)))
    plt.close(fig)
