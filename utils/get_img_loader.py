#%%
import json
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image
import os.path
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%%


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'infere')
images = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

#%%

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.ids = list(sorted(images))

    def __getitem__(self, index):
        # Image ID
        img_name = self.ids[index]
        # open the input image
        img = Image.open(os.path.join(self.root, img_name))

        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, {}

    def __len__(self):
        return len(self.ids)

def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)



def get_datasets():
    dataset = myOwnDataset(root=data_dir, transforms=get_transform())
    return dataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(batch_size):
    dataset = get_datasets()
    data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=collate_fn)
    return data_loader
# %%
