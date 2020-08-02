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


for infile in sorted(glob.glob('{}/*.png'.format(data_dir))):
    images.append(os.path.basename(infile))


frame = cv2.imread(os.path.join(data_dir, images[0]))
height, width, layers = frame.shape

out = cv2.VideoWriter(config.VIDEO_OUTOUT_FILENAME, cv2.VideoWriter_fourcc(
    *'DIVX'), 10, (width, height))


for img in images:
    out.write(cv2.imread(os.path.join(data_dir, img)))
cv2.destroyAllWindows()
out.release()
# %%
