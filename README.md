# Pynoptorch
Pytorch + panoptic segmentation 

This project implements the EfficientPS architecture for panoptic segmentation. We recommend using Hasty.ai to do your annotations. It will provide all the data needed to run Pynoptorch.

# Requirements

* Linux (tested on Ubuntu 18.04)
* conda 4.7+
* CUDA 9.0 or higher

Other requirements such as pytorch, torchvision and/or cudatoolkit will be installed when creating the conda environment from the yml file.

# Getting Started

## Init

Create temporal folders and image directories. Run from terminal:
```
./init.sh
```

## Create conda virtual environment

```
conda env create -f conda_environment.yml
```


# Data

* Put your coco annotations file in the root folder
* Place all the images under data/

Set AUTOMATICALLY_SPLIT_SETS = True in config.py the first time you run train_ignite.py  to populate data_train and data_val folders with random images taken from the data/ folder. The validation set size can be set in config.py with SPLITS.VAL_SIZE. 
If you want to keep the same training and validation sets for future training runs set AUTOMATICALLY_SPLIT_SETS = False in config.py

```
Pynoptorch
├── data
├── data_train (Automatically created)
├── data_val (Automatically created)
├── semantic_segmentation_data (put all the semantic segmentation masks here)
├── coco_hasty_annotations.json
.
.
.
```

# Train with ignite 

First, set the backbones network in config.py. Either "maskrcnn_backbone" or "EfficientNetB${0-7}"
eg,. "EfficientNetB1". Then run:

```
python train_ignite.py
```

The weights are saved every epoch under tmp/models/

# Evaluate

Set MODEL_WEIGHTS_FILENAME in config.py  eg,. "tmp/models/EfficientPS_weights_maskrcnn_backbone_loss_0.57_bbx_82.8_segm_72.0.pth". Then run:

```
python eval_coco.py
```

<!-- # Commit

To commit to this repository please follow smart commit syntax: https://support.atlassian.com/jira-software-cloud/docs/process-issues-with-smart-commits/ -->