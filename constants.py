#All dirs relative to root
DATA_LOADERS_LOC = "tmp/data_loaders/"
COCO_ANN_LOC = "tmp/coco_ann/"
MODELS_LOC = "tmp/models/"
RES_LOC = "tmp/res/"

# if AUTOMATICALLY_SPLIT_SETS is True then this dir will serve both train and val images

TRAIN_DIR = "data_train/"

#Required for evaluation unless AUTOMATICALLY_SPLIT_SETS is True
VAL_DIR = "data_val/"


DATA_LOADER_TRAIN_FILANME = "data_loader_train.pth"
DATA_LOADER_VAL_FILENAME = "data_loader_val.pth"

DATA_LOADER_TRAIN_FILENAME_BG = "data_loader_train_bg.pth"
DATA_LOADER_VAL_FILENAME_BG = "data_loader_val_bg.pth"

DATA_LOADER_TRAIN_FILENAME_OBJ = "data_loader_train_obj.pth"
DATA_LOADER_VAL_FILENAME_OBJ = "data_loader_val_obj.pth"


ANN_VAL_DEFAULT_NAME = "coco_ann_val.json"
ANN_TRAIN_DEFAULT_NAME = "coco_ann_train.json"

ANN_VAL_DEFAULT_NAME_OBJ = "coco_ann_val_obj.json"
ANN_TRAIN_DEFAULT_NAME_OBJ = "coco_ann_train_obj.json"

ANN_VAL_DEFAULT_NAME_BG = "coco_ann_val_bg.json"
ANN_TRAIN_DEFAULT_NAME_BG = "coco_ann_train_bg.json"
#EVALUATION

COCO_RES_JSON_FILENAME = "tmp/res/instances_val_obj_results.json"