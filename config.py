# All dirs relative to root
NUM_CLASSES = 10 #including background

BATCH_SIZE = 2
MODEL = "EfficientPS"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"

## Available backbone networks are: 
# "resnet50", 
# "EfficientNetB0", 
# "EfficientNetB1", 
# "EfficientNetB2",
# "EfficientNetB3", 
# "EfficientNetB4", 
# "EfficientNetB5", 
# "EfficientNetB6"
# "EfficientNetB7"
# BACKBONE = "resnet50"
BACKBONE = "EfficientNetB0"

BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 7 #excluding background
NUM_STUFF_CLASSES = 2 #excluding background

# Object tracking
MAX_DETECTIONS = 50 # Maximum number of tracked objects
NUM_FRAMES = 5 # Number of frames before recycling the ids 

ORIGINAL_INPUT_SIZE_HW = (1080, 1920) 

MIN_SIZE = 800 # Taken from maskrcnn defaults  
MAX_SIZE = 1333 # Taken from maskrcnn defaults 


# DATA = "data/"
DATA = "random_left/"
# SEMANTIC_SEGMENTATION_DATA = "semantic_segmentation_data"
SEMANTIC_SEGMENTATION_DATA = "semantic_masks_18_01"
SEMANTIC_MASKS_FORMAT = ".png"
MAX_EPOCHS = 100


# If USE_PREEXISTING_DATA_LOADERS is True new data_loaders will not be written
USE_PREEXISTING_DATA_LOADERS = False
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"
DATA_LOADER_VAL_FILENAME_OBJ = "tmp/data_loaders/data_loader_val_obj.pth"

# if USE_PREEXISTING_DATA_LOADERS is false then you can automatically split the data set
# into val and train
# Otherwise you have to split the data set manually into their respective folders:
# constants.TRAIN_DIR and constants.VAL_DIR
AUTOMATICALLY_SPLIT_SETS = True

# Set the validation set size. This makes no difference if USE_PREEXISTING_DATA_LOADERS is True
SPLITS = {"VAL_SIZE": 0.157}

# HASTY_COCO_ANN  is the coco ann file exported by hasty
# HASTY_COCO_ANN = "coco_hasty_annotations.json"
HASTY_COCO_ANN = "ann_18_01.json"

# COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# constants.COCO_ANN_LOC/constants.ANN_TRAIN_DEFAULT_NAME
# and constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME respectively
# If you want to use your own annotation files please especify the location here
# eg, "tmp/coco_ann/coco_ann_train.json"
COCO_ANN_TRAIN = None
COCO_ANN_VAL = None

# CHECKPOINT =  "tmp/models/EfficientPS_weights_resnet50_loss_0.9881418943405151.pth"
CHECKPOINT = None
# --------EVALUATION---------------

# Set the model weights to be used for evaluation
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.5556591153144836.pth"
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.8619731664657593.pth"
MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.4168185591697693.pth" # 370 samples

# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None
IOU_TYPES = ["bbox", "segm"]

# ----------INFERENCE ----------

TEST_DIR = "data_val/"

INSTANCE =  False
SEMANTIC =  True
PANOPTIC =  False

# Customize colors for semantic inference. The length of this array must be #classes + background. 
# Defaults to None. If None colors are pseudo-randomnly selected. 
# Check utils/show_segmentation.py for details
SEM_COLORS = None
# SEM_COLORS = [
#     [0/255, 0/255, 0/255],
#     [30/255, 95/255, 170/255],
#     [245/255, 208/255, 58/255],
#     [224/255, 67/255, 217/255],
#     [255/255, 255/255, 153/255],
#     [177/255, 89/255, 40/255],
#     [166/255, 3/255, 3/255],
#     [65/255, 117/255, 5/255],
#     [94/255, 245/255, 184/255],
#     [140/255, 190/255, 248/255]
# ]

# -------- REAL TIME ------
RT_INSTANCE =  False
RT_SEMANTIC =  False
RT_PANOPTIC =  True 

OBJECT_TRACKING = True
# Camera source can be either a camera device or a video
CAM_SOURCE = 0
# CAM_SOURCE = "video.avi"
SAVE_VIDEO = True
RT_VIDEO_OUTPUT_BASENAME = "rt"
RT_VIDEO_OUTPUT_FOLDER = "rt_videos/"

# ----- MAKE VIDEO --------

FPS = 5

# folder containing the images
VIDEO_CONTAINER_FOLDER = "EfficientPS2_resnet50_results_panoptic_novatron_set7/"
# video filename
VIDEO_OUTOUT_FILENAME = "panoptic_novatron_set7.avi"