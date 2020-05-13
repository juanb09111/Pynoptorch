
import cv2
import torchvision
import torch
import numpy as np

from . import show_bbox as show_bbox




def show_results(model, dataiter, num_samples, num_classes, confidence = 0.5 ):
    colors = show_bbox.get_colors_palete(num_classes)
    
    for i in range(num_samples):
        images, targets = dataiter.next()

        model.eval()
        with torch.no_grad():
            outputs= model(images)

        img = images[0].permute(1,2,0).numpy()
        img = cv2.UMat(img).get()

        show_bbox.show_bbox(img, outputs[0], confidence, colors)



