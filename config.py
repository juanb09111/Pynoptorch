# All dirs relative to root
NUM_CLASSES = 8 #including background
# NUM_CLASSES = 11
BATCH_SIZE = 1
MODEL = "EfficientPS"
MODEL_WEIGHTS_FILENAME_PREFIX = "EfficientPS_weights"
DEVICE = None #No need to assign manually

# BACKBONE = "EfficientNetB2"
BACKBONE = "resnet50"
BACKBONE_OUT_CHANNELS = 256
NUM_THING_CLASSES = 6 #excluding background
NUM_STUFF_CLASSES = 1 #excluding background

# Object tracking
MAX_DETECTIONS = 25
NUM_FRAMES = 5

# NUM_THING_CLASSES = 7
# NUM_STUFF_CLASSES = 3
# ORIGINAL_INPUT_SIZE_HW = (1200, 1920)
ORIGINAL_INPUT_SIZE_HW = (480, 640)
# ORIGINAL_INPUT_SIZE_HW = (460, 599)

MIN_SIZE = 800 # Taken from maskrcnn defaults  
MAX_SIZE = 1333 # Taken from maskrcnn defaults 
# MIN_SIZE = 400
# MAX_SIZE = 7007
# -------- TRAINING--------------
# DATA = "data_novatron/"
# SEMANTIC_SEGMENTATION_DATA = "semantic_segmentation_data_novatron"

DATA = "data/"
SEMANTIC_SEGMENTATION_DATA = "semantic_segmentation_data"
MAX_EPOCHS = 50


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
HASTY_COCO_ANN = "coco_hasty_annotations.json"
# HASTY_COCO_ANN = "annotation_novatron_16_06.json"

# COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# constants.COCO_ANN_LOC/constants.ANN_TRAIN_DEFAULT_NAME
# and constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME respectively
# If you want to use your own annotation files please especify the location here
# eg, "tmp/coco_ann/coco_ann_train.json"
COCO_ANN_TRAIN = None
COCO_ANN_VAL = None

# --------EVALUATION---------------

# Set the model weights to be used for evaluation
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_maskrcnn_backbone_loss_0.57_bbx_82.8_segm_72.0.pth"
# MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS2_weights_resnet50_loss_0.10_bbx_0.83_segm_0.85_semseg_0.74_novatron.pth"

MODEL_WEIGHTS_FILENAME = "tmp/models/EfficientPS_weights_resnet50_loss_0.5556591153144836.pth"

# Set the data loader to be used for evaluation. This can be set to None to use default filename
DATA_LOADER = None
# DATA_LOADER = "tmp/data_loaders/data_loader_train_obj.pth"
# Set coco evaluation IoU types
IOU_TYPES = ["bbox", "segm"]

# ----------TEST ----------

TEST_DIR = "data_test/seq/"

INSTANCE =  False
SEMANTIC =  False
PANOPTIC =  True

# -------- REAL TIME ------
RT_INSTANCE =  False
RT_SEMANTIC =  False
RT_PANOPTIC =  True 
CAM_SOURCE = 0
# CAM_SOURCE = "plain.avi"
SAVE_VIDEO = False
RT_VIDEO_OUTPUT_BASENAME = "rt"
RT_VIDEO_OUTPUT_FOLDER = "rt_videos/"

# ----- MAKE VIDEO --------

FPS = 5

# VIDEO_CONTAINER_FOLDER = "EfficientPS_resnet50_results_instance_1596320454.678718/"
# VIDEO_OUTOUT_FILENAME = "instance_segmentation.avi"

# VIDEO_CONTAINER_FOLDER = "EfficientPS_resnet50_results_panoptic_1596310875.777973/"
# VIDEO_OUTOUT_FILENAME = "panoptic_segmentation.avi"

VIDEO_CONTAINER_FOLDER = "EfficientPS2_resnet50_results_panoptic_novatron_set7/"
VIDEO_OUTOUT_FILENAME = "panoptic_novatron_set7.avi"