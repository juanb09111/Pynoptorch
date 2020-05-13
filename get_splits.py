from os import walk
import config
import constants
import random
import math
from shutil import copyfile, rmtree
import os


def get_splits():
    img_folder = config.DATA
    dst_train = constants.TRAIN_DIR
    dst_val = constants.VAL_DIR

    dir_filenames = []

    # Remove all files in folder
    for folder in [dst_train, dst_val]:

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    for(_, _, filenames) in walk(img_folder):
        dir_filenames.extend(filenames)

    val_length = math.floor(len(dir_filenames)*config.SPLITS["VAL_SIZE"])

    random_files = random.sample(dir_filenames, k=val_length)

    for file in dir_filenames:
        if file in random_files:
            copyfile(os.path.join(img_folder, file), os.path.join(dst_val, file))
        else:
            copyfile(os.path.join(img_folder, file), os.path.join(dst_train, file))
