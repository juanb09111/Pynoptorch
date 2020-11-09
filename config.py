# All dirs relative to root
NUM_CLASSES = 8 #including background

BATCH_SIZE = 1
MODEL = "EfficientPS"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"

BACKBONE = "resnet50" # This is the only one available at the moment
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 6 #excluding background
NUM_STUFF_CLASSES = 1 #excluding background

# Object tracking
MAX_DETECTIONS = 100 # Maximum number of tracked objects
NUM_FRAMES = 5 # Number of frames before recycling the ids 

# ORIGINAL_INPUT_SIZE_HW = (1200, 1920) 
ORIGINAL_INPUT_SIZE_HW = (375, 1242)
RESIZE = 0.5
MIN_SIZE = 800 # Taken from maskrcnn defaults  
MAX_SIZE = 1333 # Taken from maskrcnn defaults 


DATA = "data/"
# LIDAR_DATA = "lidar_data_jd/"
LIDAR_DATA = "lidar_data/"
SEMANTIC_SEGMENTATION_DATA = "semantic_segmentation_data"
SEMANTIC_MASKS_FORMAT = ".png"

MAX_EPOCHS = 50


# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = True
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"
DATA_LOADER_VAL_FILENAME_OBJ = "tmp/data_loaders/data_loader_val_obj.pth"

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
MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.5556591153144836.pth"

# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None
IOU_TYPES = ["bbox", "segm"]

# ----------INFERENCE ----------

TEST_DIR = "data_val/"

INSTANCE =  False
SEMANTIC =  False
PANOPTIC =  True




# -------- REAL TIME ------
RT_INSTANCE =  False
RT_SEMANTIC =  False
RT_PANOPTIC =  True 

# Exclude the following classes from rendering
EXCLUDE_CLASSES = ["LogPile", "Gripper", "Tractor", "Load", "Car", "Track road "]
# EXCLUDE_CLASSES= ["Tree"]

OBJECT_TRACKING = True
# Camera source can be either a camera device or a video
# CAM_SOURCE = 0
CAM_SOURCE = "plain.avi"
SAVE_VIDEO = True
RT_VIDEO_OUTPUT_BASENAME = "rt"
RT_VIDEO_OUTPUT_FOLDER = "rt_videos/To_Timo/"

# ----- MAKE VIDEO --------

FPS = 5

# folder containing the images
VIDEO_CONTAINER_FOLDER = "EfficientPS2_resnet50_results_panoptic_novatron_set7/"
# video filename
VIDEO_OUTOUT_FILENAME = "panoptic_novatron_set7.avi"


# Leevi ---------------------


# # All dirs relative to root
# NUM_CLASSES = 21 #including background

# BATCH_SIZE = 2
# MODEL = "EfficientPS"
# MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"

# BACKBONE = "resnet50" # This is the only one available at the moment
# BACKBONE_OUT_CHANNELS = 256
# NUM_THING_CLASSES = 18 #excluding background
# NUM_STUFF_CLASSES = 2 #excluding background

# # Object tracking
# MAX_DETECTIONS = 50 # Maximum number of tracked objects
# NUM_FRAMES = 5 # Number of frames before recycling the ids

# ORIGINAL_INPUT_SIZE_HW = (968, 1296)

# MIN_SIZE = 800 # Taken from maskrcnn defaults
# MAX_SIZE = 1333 # Taken from maskrcnn defaults

# DATA = "data_leevi/color_red/"
# SEMANTIC_SEGMENTATION_DATA = "data_leevi/segmentation_masks"
# SEMANTIC_MASKS_FORMAT = ".png"
# MAX_EPOCHS = 50

# # If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
# USE_PREEXISTING_DATA_LOADERS = True
# DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
# DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"
# DATA_LOADER_VAL_FILENAME_OBJ = "tmp/data_loaders/data_loader_val_obj.pth"

# # if USE_PREEXISTING_DATA_LOADERS is false then you can automatically split the data set
# # into val and train
# # Otherwise you have to split the data set manually into their respective folders:
# # constants.TRAIN_DIR and constants.VAL_DIR
# AUTOMATICALLY_SPLIT_SETS = False

# # Set the validation set size. This makes no difference if USE_PREEXISTING_DATA_LOADERS is True
# SPLITS = {"VAL_SIZE": 0.3}

# # HASTY_COCO_ANN  is the coco ann file exported by hasty
# HASTY_COCO_ANN = "data_leevi/annotations_red.json"

# # COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# # constants.COCO_ANN_LOC/constants.ANN_TRAIN_DEFAULT_NAME
# # and constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME respectively
# # If you want to use your own annotation files please especify the location here
# # eg, "tmp/coco_ann/coco_ann_train.json"
# COCO_ANN_TRAIN = None
# COCO_ANN_VAL = None

# # --------EVALUATION---------------

# # Set the model weights to be used for evaluation
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.5556591153144836.pth"

# # Set the data loader to be used for evaluation. This can be set to None to use default filename
# DATA_LOADER = None
# IOU_TYPES = ["bbox", "segm"]

# # ----------INFERENCE ----------

# TEST_DIR = "data_val/"

# INSTANCE =  False
# SEMANTIC =  False
# PANOPTIC =  True

# # -------- REAL TIME ------
# RT_INSTANCE =  False
# RT_SEMANTIC =  False
# RT_PANOPTIC =  True

# OBJECT_TRACKING = True
# # Camera source can be either a camera device or a video
# CAM_SOURCE = 0
# # CAM_SOURCE = "video.avi"
# SAVE_VIDEO = True
# RT_VIDEO_OUTPUT_BASENAME = "rt"
# RT_VIDEO_OUTPUT_FOLDER = "rt_videos/"

# # ----- MAKE VIDEO --------

# FPS = 5

# # folder containing the images
# VIDEO_CONTAINER_FOLDER = "EfficientPS2_resnet50_results_panoptic_novatron_set7/"
# # video filename
# VIDEO_OUTOUT_FILENAME = "panoptic_novatron_set7.avi"
