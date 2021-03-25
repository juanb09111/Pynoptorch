# %%
import re
import os.path
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import glob
import config

data_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../{}'.format(config.VIDEO_CONTAINER_FOLDER))

images = []

def get_number(file_name):

    # f_name= file_name.split("/")[-1].split("-")[-1]
    f_name= file_name.split("/")[-1].split(".")[0].split("_")[0]
    f_name = int(f_name) + 1000
    f_name = str(f_name)
    return f_name

for infile in sorted(glob.glob('{}/*.png'.format(data_dir)), key=get_number):
    images.append(os.path.basename(infile))


frame = cv2.imread(os.path.join(data_dir, images[0]))
height, width, layers = frame.shape

out = cv2.VideoWriter(config.VIDEO_OUTOUT_FILENAME, cv2.VideoWriter_fourcc(
    *'DIVX'), config.FPS, (width, height))


for img in images:
    out.write(cv2.imread(os.path.join(data_dir, img)))
cv2.destroyAllWindows()
out.release()
# %%
