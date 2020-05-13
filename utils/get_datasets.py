# %%
import json
import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image

# %%
f = open(os.path.dirname(__file__) + '/../label_classes.json')
data = json.load(f)
categories = list(map(lambda x: x['class_name'], data))

# %%


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        category_ids = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = coco_annotation[i]['category_id']
            label = coco.cats[category_id]['name']
            labels.append(categories.index(label) + 1)

            area = coco_annotation[i]['bbox'][2] * \
                coco_annotation[i]['bbox'][3]
            areas.append(area)

            iscrowd.append(coco_annotation[i]['iscrowd'])

            mask = coco.annToMask(coco_annotation[i])
            masks.append(mask)

            category_ids.append(category_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Iscrowd
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation['masks'] = masks
        my_annotation["category_ids"] = category_ids

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(custom_transforms)

# path to your own data and coco file


# create own Dataset

def get_datasets(root, annotation, split = False, val_size=0.20):
    

    my_dataset = myOwnDataset(root=root,
                              annotation=annotation,
                              transforms=get_transform()
                              )
    if split:
        if val_size >= 1:
            raise AssertionError("val_size must be a value in the range of (0,1)")

        len_val = math.ceil(len(my_dataset)*val_size)
        len_train = len(my_dataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")
        train_set, val_set = torch.utils.data.random_split(
            my_dataset, [len_train, len_val])
        return train_set, val_set
    else:
        return my_dataset


def collate_fn(batch):
    return tuple(zip(*batch))



def get_dataloaders(batch_size, root, annotation, split = False, val_size=0.20):
    if split:
        train_set, val_set = get_datasets(root=root,
                                        annotation=annotation, split= split, val_size = val_size)
        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        collate_fn=collate_fn,
                                                        drop_last=True)

        data_loader_val = torch.utils.data.DataLoader(val_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=collate_fn,
                                                    drop_last=True)
        return data_loader_train, data_loader_val
    else:
        dataset = get_datasets(root=root, annotation=annotation)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=collate_fn,
                                                    drop_last=True)
        return data_loader
