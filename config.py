# All dirs relative to root
NUM_CLASSES = 8
BATCH_SIZE = 1
MODEL = "EfficientPS"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"
DEVICE = None

# BACKBONE = "EfficientNetB2"
BACKBONE = "resnet50"
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 6
NUM_STUFF_CLASSES = 1
ORIGINAL_INPUT_SIZE_HW = (1200, 1920)

# -------- TRAINING--------------
DATA = "data/"
SEMANTIC_SEGMENTATION_DATA = "semantic_segmentation_data"
MAX_EPOCHS = 100


# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = False
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"

# if USE_PREEXISTING_DATA_LOADERS is false then you can automatically split the data set
# into val and train
# Otherwise you have to split the data set manually into their respective folders:
# constants.TRAIN_DIR and constants.VAL_DIR
AUTOMATICALLY_SPLIT_SETS = False

# Set the validation set size. This makes no difference if USE_PREEXISTING_DATA_LOADERS is True
SPLITS = {"VAL_SIZE": 0.157}

# HASTY_COCO_ANN  is the coco ann file exported by hasty
HASTY_COCO_ANN = "coco_hasty_annotations.json"

# COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# constants.COCO_ANN_LOC/constants.ANN_TRAIN_DEFAULT_NAME
# and constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME respectively
# If you want to use your own annotation files please especify the location here
# eg, "tmp/coco_ann/coco_ann_train.json"
COCO_ANN_TRAIN = None
COCO_ANN_VAL = None

# --------EVALUATION---------------

# Set the model weights to be used for evaluation
MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_maskrcnn_backbone_loss_0.57_bbx_82.8_segm_72.0.pth"
# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None
# DATA_LOADER = "tmp/data_loaders/data_loader_train_obj.pth"
# Set coco evaluation IoU types
IOU_TYPES = ["bbox", "segm"]

# ----------TEST ----------

INSTANCE =  False
SEMANTIC =  False 
PANOPTIC =  False 

# ----- MAKE VIDEO --------

# VIDEO_CONTAINER_FOLDER = "EfficientPS_resnet50_results_instance_1596320454.678718/"
# VIDEO_OUTOUT_FILENAME = "instance_segmentation.avi"

# VIDEO_CONTAINER_FOLDER = "EfficientPS_resnet50_results_panoptic_1596310875.777973/"
# VIDEO_OUTOUT_FILENAME = "panoptic_segmentation.avi"

VIDEO_CONTAINER_FOLDER = "EfficientPS_resnet50_results_semantic_1596374348.640805/"
VIDEO_OUTOUT_FILENAME = "semantic_segmentation_dewd.avi"