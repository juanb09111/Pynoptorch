# %%
import scipy.io
import os.path
import math
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np

import config_jd_lidar

import cv2
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
# %%




class jdDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_root, imPts_root, lidar_fov_root, sparse_depth_root, transforms=None, n_samples=None):

        self.imgs_root = imgs_root
        self.imPts_root = imPts_root
        self.lidar_fov_root = lidar_fov_root
        self.sparse_depth_root = sparse_depth_root

        self.file_names = [f for f in listdir(imgs_root) if isfile(join(imgs_root, f))]
        self.transforms = transforms

    def visualize_projection(self, img_filename, imPts_crop, upper, lower, left, right):
        #read and crop image
        img = cv2.imread(img_filename)[int(upper):int(lower), int(left):int(right)]
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        # print("shape", imPts_crop.shape)
        for i in range(imPts_crop.shape[0]):
            depth = imPts_crop[i, 2]
            color = cmap[int(640.0 / depth), :]
            cv2.circle(rgb, (int(np.round(imPts_crop[i, 0])),
                             int(np.round(imPts_crop[i, 1]))),
                       2, color=tuple(color), thickness=-1)
        plt.imshow(rgb)
        plt.yticks([])
        plt.xticks([])
        plt.show()
        
        # plt.savefig('projectionjd.png')

    def pre_process_points(self, imPts, lidar_fov):
        #round imtPts
        round_imPts = torch.round(imPts)
        
        _, indices = torch.unique(round_imPts[:, 0:2], dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        round_imPts_unique = round_imPts[unique_indices]
        round_imPts_unique[:, 2] = imPts[unique_indices, 2]
        lidar_fov_unique = lidar_fov[unique_indices]

        rand_perm = torch.randperm(round_imPts_unique.shape[0])
        round_imPts_unique = round_imPts_unique[rand_perm, :]
        lidar_fov_unique = lidar_fov_unique[rand_perm, :]
        
        return round_imPts_unique[:config_jd_lidar.N_NUMBER, :], lidar_fov_unique[:config_jd_lidar.N_NUMBER, :]

    def find_k_nearest(self, lidar_fov):
        k_number = config_jd_lidar.K_NUMBER
        b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

        distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
        _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # B x N x 3

        return indices.squeeze_(0).long()

    def __getitem__(self, index):

        img_filename = os.path.join(self.imgs_root, self.file_names[index])  # rgb image

        basename = img_filename.split("/")[-1].split(".")[0]

        imPts_filename = os.path.join(self.imPts_root, basename+".mat")
        lidar_fov_filename = os.path.join(self.lidar_fov_root, basename+".mat")
        sparse_depth_filename = os.path.join(self.sparse_depth_root, basename+".mat")
        # print("filename", basename,  img_filename, imPts_filename)
        
        imPts = scipy.io.loadmat(imPts_filename)["imPts"]
        lidar_fov = scipy.io.loadmat(lidar_fov_filename)["lidar_fov"]
        sparse_depth = scipy.io.loadmat(sparse_depth_filename)["sparse_depth"]

        imPts = torch.tensor(imPts)
        lidar_fov = torch.tensor(lidar_fov)
        sparse_depth = torch.tensor(sparse_depth)
        
        # get img
        img = Image.open(img_filename)

        # img width and height
        (img_width, img_height) = (img.width, img.height)

        if self.transforms is not None:
            img = self.transforms(crop=True)(img)
        

        # remove duplicated points and shuffle

        imPts, lidar_fov = self.pre_process_points(imPts, lidar_fov)

        # crop

        upper = np.ceil((img_height - config_jd_lidar.CROP_OUTPUT_SIZE[0])/2)
        lower = np.floor(
            (img_height - config_jd_lidar.CROP_OUTPUT_SIZE[0])/2) + config_jd_lidar.CROP_OUTPUT_SIZE[0]

        left = np.ceil((img_width - config_jd_lidar.CROP_OUTPUT_SIZE[1])/2)
        right = np.floor(
            (img_width - config_jd_lidar.CROP_OUTPUT_SIZE[1])/2) + config_jd_lidar.CROP_OUTPUT_SIZE[1]

        inds = np.where((torch.round(imPts[:,0]) < right - 1) & (torch.round(imPts[:,0]) > left + 1) & (torch.round(imPts[:,0]) >= 0) &
                        (torch.round(imPts[:,1]) < lower - 1) & (torch.round(imPts[:,1]) > upper + 1) & (torch.round(imPts[:,1]) >= 0) &
                        (imPts[:, 2] > 0)
                        )[0]

        # move to origin
        imPts[:, 0] = imPts[:, 0] - left
        imPts[:, 1] = imPts[:, 1] - upper

        
        # select indices
        imPts_crop = torch.tensor(imPts[inds, :])
     
        lidar_fov_crop = torch.tensor(lidar_fov[inds, :], dtype=torch.float)


        # self.visualize_projection(img_filename, imPts_crop, upper, lower, left, right)

        #get knn indices
        k_nn_indices = self.find_k_nearest(lidar_fov_crop)

        #get mask
        mask = torch.zeros(img.shape[1:], dtype=torch.bool)

        # pixel coor
        y_coor= torch.tensor(torch.round(imPts_crop[:, 1]), dtype=torch.long)
        x_coor= torch.tensor(torch.round(imPts_crop[:, 0]), dtype=torch.long)
        mask[y_coor, x_coor] = True
        # maskre = mask.view(1, -1)
        # print("mask", maskre.shape, (maskre == True).nonzero(as_tuple=True)[0].shape)

        sparse_depth_crop = torch.zeros_like(img[0, :, :].unsqueeze_(0), dtype=torch.float)
        sparse_depth_crop[0, y_coor, x_coor] = torch.tensor(imPts_crop[:, 2], dtype=torch.float)
        
        return img, lidar_fov_crop, mask, sparse_depth_crop, k_nn_indices, basename

    def __len__(self):
        return len(self.file_names)



def get_transform(resize=False, normalize=False, crop=False):
    new_size = tuple(np.ceil(x*config_jd_lidar.RESIZE)
                     for x in config_jd_lidar.ORIGINAL_INPUT_SIZE_HW)
    new_size = tuple(int(x) for x in new_size)
    custom_transforms = []
    if resize:
        custom_transforms.append(transforms.Resize(new_size))

    if crop:
        custom_transforms.append(
            transforms.CenterCrop(config_jd_lidar.CROP_OUTPUT_SIZE))

    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)/255).unsqueeze(0)))
    if normalize:
        custom_transforms.append(transforms.Normalize(0.485, 0.229))
    return transforms.Compose(custom_transforms)




def get_datasets(imgs_root, imPts_root, lidar_fov_root, sparse_depth_root, split=False, val_size=0.20, n_samples=None):
    
    jd_dataset = jdDataset(imgs_root=imgs_root, imPts_root=imPts_root, lidar_fov_root=lidar_fov_root, sparse_depth_root=sparse_depth_root, transforms=get_transform, n_samples=n_samples)

    jd_dataset.__getitem__(0)
    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

        len_val = math.ceil(len(jd_dataset)*val_size)
        len_train = len(jd_dataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")
        train_set, val_set = torch.utils.data.random_split(
            jd_dataset, [len_train, len_val])
        return train_set, val_set
    else:
        return jd_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size, imgs_root, imPts_root, lidar_fov_root, sparse_depth_root, split=False, val_size=0.20, n_samples=None):
    
    
    dataset = get_datasets(imgs_root=imgs_root, imPts_root=imPts_root, lidar_fov_root=lidar_fov_root, sparse_depth_root=sparse_depth_root, n_samples=n_samples)

    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                collate_fn=collate_fn,
                                                drop_last=True)
    return data_loader


# imgs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data_jd_lidar/data_jd_lidar/imgs/")
# imPts_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data_jd_lidar/data_jd_lidar/imPts/")
# lidar_fov_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data_jd_lidar/data_jd_lidar/lidar_fov/")
# sparse_depth_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data_jd_lidar/data_jd_lidar/sparse_depth/")

# kitti_data_loader=get_dataloaders(config_jd_lidar.BATCH_SIZE, imgs_root, imPts_root, lidar_fov_root, sparse_depth_root, split=False, val_size=0.20, n_samples=None)
