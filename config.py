# All dirs relative to root

# ----------SET UP MODEL-------------------------- 

NUM_THING_CLASSES = 7 #excluding background
NUM_STUFF_CLASSES = 2 #excluding background


BATCH_SIZE = 1
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
BACKBONE = "resnet50"

# USE DEPTHWISE SEPARABLE CONVOLUTIONS INSTEAD OF TRADITIONAL CONVOLUTIONAL LAYERS ? https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
BACKBONE_DEPTHWISE_CONV = True
SEMANTIC_HEAD_DEPTHWISE_CONV = True

# 
BACKBONE_OUT_CHANNELS = 256

ORIGINAL_INPUT_SIZE_HW = (1080, 1920) 
# ORIGINAL_INPUT_SIZE_HW = (540, 960)
MIN_SIZE = 800 # Taken from maskrcnn defaults  
MAX_SIZE = 1333 # Taken from maskrcnn defaults 



#--------------TRAINING SETTINGS--------------------------------

# Maximum number of training epochs
MAX_EPOCHS = 100


# Training data location
DATA = "data/"

# Data augmentation:
# It is possible to use an extra training data set corresponding to a augmentation of the original training set. 
# e.g. https://github.com/NVIDIA/FastPhotoStyle/blob/master/TUTORIAL.md
# In order to do so, please name your augmented or stylized images according to the following format:

# Original source image name: {original_name} 
# Augmented/stylized image name: {original_name} + "_augmented"

#e.g.
# Original source image name: "my_image.jpg" 
# Augmented/stylized image name: "my_image_augmented.jpg"

# Location
AUGMENTED_DATA_LOC = "augmented_data/"

AUGMENTED_PERCENT = 0.5 # number between 0 and 1 corresponding to the number of augmented images to use for training with respect to 
# the number of images in the original training dataset so that the total number of images use for training is then: n_original + n_original*AUGMENTED_PERCENT

SEMANTIC_SEGMENTATION_DATA_LOC = "semantic_segmentation_data"
SEMANTIC_MASKS_FORMAT = ".png"


USE_TORCHVISION_AUGMENTATION = False

# If USE_PREEXISTING_DATA_LOADERS is False, then new data_loaders will be created.
USE_PREEXISTING_DATA_LOADERS = False
# Otherwise, if USE_PREEXISTING_DATA_LOADERS is True please state the location of the dataloader to be used for training here:
DATA_LOADER_TRAIN_FILANME = "tmp/data_loaders/data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "tmp/data_loaders/data_loader_val.pth"
DATA_LOADER_VAL_FILENAME_OBJ = "tmp/data_loaders/data_loader_val_obj.pth"


# if USE_PREEXISTING_DATA_LOADERS is false then you can automatically split the dataset
# into val and train
# Otherwise you have to split the dataset manually into their respective folders:
# constants.TRAIN_DIR and constants.VAL_DIR
AUTOMATICALLY_SPLIT_SETS = False

# Set the validation set size. This makes no difference if USE_PREEXISTING_DATA_LOADERS is True
SPLITS = {"VAL_SIZE": 0.20}

# HASTY_COCO_ANN  is the coco ann file exported by hasty
HASTY_COCO_ANN = "coco_hasty_annotations.json"


# COCO_ANN_TRAIN and COCO_ANN_VAL are created and saved automatically according to
# {constants.COCO_ANN_LOC}/{constants.ANN_TRAIN_DEFAULT_NAME}
# and {constants.COCO_ANN_LOC/constants.ANN_VAL_DEFAULT_NAME} respectively
# If you want to use your own annotation files please especify the location here
# eg, "tmp/coco_ann/coco_ann_train.json" otherwise set them as None
COCO_ANN_TRAIN = None
COCO_ANN_VAL = None


# Load pre-trained weights for training. This can be set to None
# CHECKPOINT = "tmp/models/EfficientPS_weights_resnet50_loss_1.9322376251220703.pth"
CHECKPOINT = None

# --------EVALUATION---------------

# Set the model weights to be used for evaluation

MODEL_WEIGHTS_FILENAME =  "tmp/models/EfficientPS_weights_resnet50_loss_0.8475958108901978.pth" # 370 samples

# Set the data loader to be used for evaluation. This can be set to None to use default dataloader found under tmp/data_loaders/data_loader_val_obj.pth
DATA_LOADER = None
IOU_TYPES = ["bbox", "segm"]

# ----------INFERENCE ----------

TEST_DIR = "data_val/"
INFERENCE_OBJECT_TRACKING =  True

LUCAS_KANADE_FIND_KEYPOINTS = False
INFERENCE_CONFIDENCE = 0.5
INSTANCE =  False
SEMANTIC =  True
PANOPTIC =  False

# Customize colors for semantic inference. The length of this array must be n_classes + background. 
# Defaults to None. If None, colors are pseudo-randomnly selected. 
# Check utils/show_segmentation.py for details
# SEM_COLORS = None
SEM_COLORS = [
    [0/255, 0/255, 0/255],
    [30/255, 95/255, 170/255],
    [245/255, 208/255, 58/255],
    [224/255, 67/255, 217/255],
    [255/255, 255/255, 153/255],
    [177/255, 89/255, 40/255],
    [166/255, 3/255, 3/255],
    [65/255, 117/255, 5/255],
    [94/255, 245/255, 184/255],
    [140/255, 190/255, 248/255]
]



# -------- REAL TIME INFERENCE ------

# Select one of the following 3 options
RT_INSTANCE =  True
RT_SEMANTIC =  False
RT_PANOPTIC =  False 

BOUNDING_BOX_ONLY = True # If True, only bounding boxes will be shown

CONFIDENCE_THRESHOLD = 0.8 #Show predictions over this value

# Object tracking
OBJECT_TRACKING = True
MAX_DETECTIONS = 50 # Maximum number of tracked objects
NUM_FRAMES = 5 # Number of frames before recycling the tracking ids 
NMS_THRESHOLD = 0.3 # remove overlapping bounding boxes
OBJECT_TRACKING_IOU_THRESHHOLD = 0.2

SUPER_CLASS = ["car", "truck", "bike", "bus"] # When tracking objects, all objects belonging to this class will be tracked as one class

# Camera source can be either a camera device or a video
# CAM_SOURCE = 0
CAM_SOURCE = "rt_videos/source/jd_nov_full_res_2.avi"
SAVE_VIDEO = True
RT_VIDEO_OUTPUT_BASENAME = "jd_nov_full_res"
RT_VIDEO_OUTPUT_FOLDER = "rt_videos/"

# ----- MAKE VIDEO --------

FPS = 5

# folder containing the images
VIDEO_CONTAINER_FOLDER = "video_images/"
# video filename
VIDEO_OUTOUT_FILENAME = "jd_nov_full_res.avi"