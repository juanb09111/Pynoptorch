# %%
import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import config
import re
# %%


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, semantic_masks_folder=None, aug_data_root=None):

        self.root = root
        self.aug_root = aug_data_root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.semantic_masks_folder = semantic_masks_folder
        catIds = self.coco.getCatIds()
        categories = self.coco.loadCats(catIds)
        self.categories = list(map(lambda x: x['name'], categories))

        self.bg_categories_ids = self.coco.getCatIds(supNms="background")
        bg_categories = self.coco.loadCats(self.bg_categories_ids)
        self.bg_categories = list(map(lambda x: x['name'], bg_categories))

        self.obj_categories_ids = self.coco.getCatIds(supNms="object")
        obj_categories = self.coco.loadCats(self.obj_categories_ids)
        self.obj_categories = list(map(lambda x: x['name'], obj_categories))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get object annotations ids from coco
        obj_ann_ids = coco.getAnnIds(
            imgIds=img_id, catIds=self.obj_categories_ids)
        # Dictionary: target coco_annotation file for an image containing only object classes
        coco_annotation = coco.loadAnns(obj_ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']

        # open the input image
        if "augmented" in path:
            if self.aug_root == None:
                raise AssertionError(
                    "augmented data root folder is not defined. Please set it in config.py")
            else:

                img = Image.open(os.path.join(self.aug_root, path))
                # remove "_augmented"
                path = re.sub('\_augmented', '', path)
        else:
            img = Image.open(os.path.join(self.root, path))

        semantic_mask_path = os.path.splitext(
            path)[0] + config.SEMANTIC_MASKS_FORMAT
        # create semantic mask
        if self.semantic_masks_folder is not None:
            semantic_mask = Image.open(os.path.join(
                self.semantic_masks_folder, semantic_mask_path))
            semantic_mask = np.array(semantic_mask)
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
            labels.append(self.obj_categories.index(label) + 1)
            # TODO: Coco does not calculate area like this, This is only a quick fix for hasty anns area=0
            area = coco_annotation[i]['bbox'][2] * \
                coco_annotation[i]['bbox'][3]
            areas.append(area)

            iscrowd.append(coco_annotation[i]['iscrowd'])

            mask = coco.annToMask(coco_annotation[i])
            masks.append(mask)

            category_ids.append(category_id)

        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            areas = torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]))
            labels = torch.zeros((1), dtype=torch.int64)
            masks = torch.zeros(
                (1, *config.ORIGINAL_INPUT_SIZE_HW), dtype=torch.uint8)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Iscrowd

        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # Num of instance objects
        num_objs = torch.as_tensor(num_objs, dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        my_annotation["category_ids"] = category_ids
        my_annotation["num_instances"] = num_objs
        my_annotation['masks'] = masks

        if self.semantic_masks_folder is not None:
            semantic_mask = torch.as_tensor(semantic_mask, dtype=torch.uint8)
            my_annotation["semantic_mask"] = semantic_mask

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


class testDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, return_next=False):
        self.root = root
        self.transforms = transforms
        self.file_names_arr = sorted(os.listdir(root))
        self.return_next = return_next

    def __getitem__(self, index):
        
        num_samples = len(self.file_names_arr)

        img = Image.open(os.path.join(self.root, self.file_names_arr[index]))

        annotation = {}
        annotation["file_name"] = self.file_names_arr[index]

        if self.return_next:
            if index+1 < num_samples:
                next_img = Image.open(os.path.join(self.root, self.file_names_arr[index+1]))

                next_annotation = {}
                next_annotation["file_name"] = self.file_names_arr[index+1]
            else:
                next_img = None
                next_annotation = None
            

        if self.transforms is not None:
            img = self.transforms(img)

        if self.transforms is not None and self.return_next and next_img is not None:
            next_img = self.transforms(next_img)
        
        if self.return_next:
            return img, next_img, annotation, next_annotation


        return img, annotation

    def __len__(self):
        return len(self.file_names_arr)


def get_transform(use_augmentation=False):
    if use_augmentation:
        custom_transforms = []
        custom_transforms.append(transforms.RandomHorizontalFlip())
        custom_transforms.append(transforms.ToTensor())
        # custom_transforms.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        custom_transforms.append(transforms.RandomErasing())
        return transforms.Compose(custom_transforms)

    else:
        custom_transforms = []
        custom_transforms.append(transforms.ToTensor())
        return transforms.Compose(custom_transforms)


# create own Dataset

def get_datasets(root, annotation=None, split=False, val_size=0.20, semantic_masks_folder=None, is_test_set=False, use_augmentation=False, aug_data_root=None, return_next=False):

    if is_test_set:
        test_dataset = testDataset(
            root, transforms=get_transform(), return_next=return_next)
        return test_dataset

    my_dataset = myOwnDataset(root=root,
                              annotation=annotation,
                              transforms=get_transform(
                                  use_augmentation=use_augmentation),
                              semantic_masks_folder=semantic_masks_folder,
                              aug_data_root=aug_data_root
                              )
    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

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


def get_dataloaders(batch_size, root, annotation=None, split=False, val_size=0.20, semantic_masks_folder=None, is_test_set=False, use_augmentation=False, aug_data_root=None, return_next=False):

    if is_test_set:
        test_set = get_datasets(
            root, is_test_set=is_test_set, use_augmentation=use_augmentation, aug_data_root=aug_data_root, return_next=return_next)
        data_loader_test = torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       collate_fn=collate_fn,
                                                       drop_last=True)
        return data_loader_test
    if split:
        train_set, val_set = get_datasets(root=root,
                                          annotation=annotation,
                                          split=split, val_size=val_size,
                                          semantic_masks_folder=semantic_masks_folder, use_augmentation=use_augmentation, aug_data_root=aug_data_root)
        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=4,
                                                        collate_fn=collate_fn,
                                                        drop_last=True)

        data_loader_val = torch.utils.data.DataLoader(val_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      collate_fn=collate_fn,
                                                      drop_last=True)
        return data_loader_train, data_loader_val
    else:
        dataset = get_datasets(
            root=root, annotation=annotation, semantic_masks_folder=semantic_masks_folder, use_augmentation=use_augmentation, aug_data_root=aug_data_root)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=collate_fn,
                                                  drop_last=True)
        return data_loader
