# All dirs relative to root
NUM_CLASSES = 8
BATCH_SIZE = 2
MODEL = "EfficientPS"
# MODEL = "MaskRCNN"
# MODEL = "DeepLab"
# MODEL_WEIGHTS_FILENAME_PREFIX = "deepLab_weights"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"
DEVICE = None
# -------- TRAINING--------------
DATA = "data/"
MAX_EPOCHS = 25


# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = True
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"

# DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train_bg.pth"
# DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val_bg.pth"

# if USE_PREEXISTING_DATA_LOADERS is false then you can automatically split the data set
# into val and train
# Otherwise you have to split the data set manually into their respective folders:
# constants.TRAIN_DIR and constants.VAL_DIR
AUTOMATICALLY_SPLIT_SETS = False

# Set the validation set size. This makes no difference if USE_PREEXISTING_DATA_LOADERS is True
SPLITS = {"VAL_SIZE": 0.157}

# HASTY_COCO_ANN  is the coco ann file exported by hasty
HASTY_COCO_ANN = "ann.json"

# COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# constants.COCO_ANN_LOC/constants.ANN_TRAIN_DEFAULT_NAME
# and constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME respectively
# If you want to use your own annotation files please especify the location here
# eg, "tmp/coco_ann/coco_ann_train.json"
COCO_ANN_TRAIN = None
COCO_ANN_VAL = None

# --------EVALUATION---------------

# Set the model weights to be used for evaluation
MODEL_WEIGHTS_FILENAME = "tmp/models/maskRCNN_weights_bg_precision_0.53_recall_0.81.pth"
# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None
# DATA_LOADER = "tmp/data_loaders/data_loader_val_bg.pth"
# Set coco evaluation IoU types
IOU_TYPES = ["bbox", "segm"]
