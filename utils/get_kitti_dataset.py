# %%
import scipy.io
import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import config
from utils.lidar_cam_projection import read_calib_file, load_velo_scan, full_project_velo_to_cam2, project_to_image
import glob
import cv2
import matplotlib.pyplot as plt
# %%


def getListOfImgs(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    sync_folders = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):

            current_path = fullPath.split("/")[-1]

            if fullPath.split("/")[-1].find("sync") != -1:
                sync_folders.append(current_path)

            new_files, new_folders = getListOfImgs(fullPath)
            sync_folders = sync_folders + new_folders
            allFiles = allFiles + new_files
        elif fullPath.find("image_02") != -1 and fullPath.find(".png") != -1:
            allFiles.append(fullPath)

    return allFiles, sync_folders


def getListOfDepthData(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfDepthData(fullPath)
        elif fullPath.find("image_02") != -1 and fullPath.find(".png") != -1:
            allFiles.append(fullPath)

    return allFiles


def get_lidar_filenames(root_folder, sync_folder, matched_imgs):

    full_path = os.path.join(root_folder, sync_folder,
                             "velodyne_points", "data")

    file_paths = glob.glob(full_path+"/*.bin")

    matched_imgs_name = [s.split("/")[-1].split(".")[0] for s in matched_imgs]

    file_paths = [s for s in file_paths if s.split(
        "/")[-1].split(".")[0] in matched_imgs_name]

    return file_paths


def get_depth_data(imgs_root, depth_velodyne_folder, depth_annotated_folder, sync_folders, imgs_list):

    depth_velo_list = list()
    depth_anotated_list = list()
    source_imgs = list()
    lidar_file_list = list()

    # list of lists
    data_lists = list((depth_velo_list, depth_anotated_list))

    # root folders for sparse lidar and ground truth
    data_depth_velodyne = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), depth_velodyne_folder)

    data_depth_annotated = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), depth_annotated_folder)

    # iterate over root folders
    for idx, data_folder in enumerate((data_depth_velodyne, data_depth_annotated)):

        # Get sync folders from sparse lidar root folder and annotations folder
        directories = glob.glob(data_folder+"*/")
        directories = list(
            map(lambda folder_path: folder_path.split("/")[-2], directories))

        # loop over sync folder from imgs folder
        for folder in sync_folders:
            # if the current folder is in directories
            if folder in directories:
                # add png files under data_folder + folder
                data_lists[idx] = data_lists[idx] + \
                    getListOfDepthData(data_folder + folder)

                # do it once for the current folder
                if idx == 0:
                    # Get images that are under the same folder in imgs
                    matching_imgs = [s for s in imgs_list if folder in s]
                    # get depth filenames that are under the current folder
                    depth_img_filenames = [
                        s for s in data_lists[idx] if folder in s]
                    # get the imgs names without basename
                    depth_img_filenames = list(
                        map(lambda file_path: file_path.split("/")[-1], depth_img_filenames))

                    # Filter images so that they are in both imgs folder and lidar data folder
                    matched_imgs = [s for s in matching_imgs if s.split(
                        "/")[-1] in depth_img_filenames]

                    # map matched imgs
                    # print(folder)
                    matched_imgs_filenames = list(
                        map(lambda file_path: file_path.split("/")[-1].split(".")[0], matched_imgs))
                    # print(matched_imgs_filenames)
                    matched_lidar_files = get_lidar_filenames(
                        imgs_root, folder, matched_imgs_filenames)

                    lidar_file_list = lidar_file_list + matched_lidar_files
                    source_imgs = source_imgs + matched_imgs

    return data_lists, source_imgs, lidar_file_list


def get_file_lists(imgs_root, data_depth_velodyne_root, data_depth_annotated_root):

    imgs, sync_folders = getListOfImgs(imgs_root)
    (data_velo, data_ann), source_imgs, lidar_file_list = get_depth_data(
        imgs_root, data_depth_velodyne_root, data_depth_annotated_root, sync_folders, imgs)

    return np.sort(source_imgs), np.sort(data_velo), np.sort(data_ann), np.sort(lidar_file_list)


class kittiDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, transforms=None):

        self.imgs_root = imgs_root
        self.data_depth_velodyne_root = data_depth_velodyne_root
        self.data_depth_annotated_root = data_depth_annotated_root
        self.calib_velo2cam = calib_velo2cam
        self.calib_cam2cam = calib_cam2cam

        source_imgs, data_velo_files, data_ann_files, lidar_files = get_file_lists(
            imgs_root, data_depth_velodyne_root, data_depth_annotated_root)

        self.source_imgs = source_imgs
        self.data_velo_files = data_velo_files
        self.data_ann_files = data_ann_files
        self.lidar_files = lidar_files
        self.transforms = transforms

    def visualize_projection(self, img_filename, imgfov_pc_pixel, imgfov_pc_cam2):

        rgb = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        for i in range(imgfov_pc_pixel.shape[1]):
            depth = imgfov_pc_cam2[2, i]
            color = cmap[int(640.0 / depth), :]
            cv2.circle(rgb, (int(np.round(imgfov_pc_pixel[0, i])),
                             int(np.round(imgfov_pc_pixel[1, i]))),
                       2, color=tuple(color), thickness=-1)
        plt.imshow(rgb)
        plt.yticks([])
        plt.xticks([])
        plt.show()

    def pre_process_points(self, imPts, lidar_fov):

        _, indices = torch.unique(imPts, dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        imPts = imPts[unique_indices]
        lidar_fov = lidar_fov[unique_indices]

        return imPts, lidar_fov

    def __getitem__(self, index):

        img_filename = self.source_imgs[index]
        lidar_filename = self.lidar_files[index]
        depth_gt = self.data_ann_files[index]

        calib_velo2cam = read_calib_file(self.calib_velo2cam)
        calib_cam2cam = read_calib_file(self.calib_cam2cam)

        img = Image.open(img_filename)
        gt_img = Image.open(depth_gt)

        pc_velo = load_velo_scan(lidar_filename)[:, :3]

        # img width and height
        (img_width, img_height) = (img.width, img.height)

        # projection matrix (project from velo2cam2)
        proj_velo2cam2 = full_project_velo_to_cam2(
            calib_velo2cam, calib_cam2cam)

        # apply projection
        pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

        # Filter lidar points to be within image FOV
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                        (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                        (pc_velo[:, 0] > 0)
                        )[0]

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d[:, inds]

        # Retrieve depth from lidar
        imgfov_pc_velo = pc_velo[inds, :]
        imgfov_pc_velo = np.hstack(
            (imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()

        # visualize
        # self.visualize_projection(
        #     img_filename, imgfov_pc_pixel, imgfov_pc_cam2)

        # to return
        imPts = torch.tensor(
            imgfov_pc_pixel, dtype=torch.float).permute(1, 0)  # N x 2
        imPts = torch.floor(imPts*config.RESIZE)  # N x 2 resized

        lidar_fov = torch.tensor(
            imgfov_pc_cam2, dtype=torch.float).permute(1, 0)  # N x 3

        # remove duplicate
        imPts, lidar_fov = self.pre_process_points(imPts, lidar_fov)

        if self.transforms is not None:
            img = self.transforms(resize=True)(img)
            gt_img = self.transforms(resize=False)(gt_img)
        print("here")
        return img, imPts, lidar_fov, gt_img

    def __len__(self):
        return len(self.source_imgs)


def get_transform(resize=True):
    new_size = tuple(np.ceil(x*config.RESIZE)
                     for x in config.ORIGINAL_INPUT_SIZE_HW)
    new_size = tuple(int(x) for x in new_size)
    custom_transforms = []
    if resize:
        custom_transforms.append(transforms.Resize(new_size))
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)


# kitti_dataset = kittiDataset(
#     imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, transforms=get_transform)

# img, imPts, lidar_fov, gt_img = kitti_dataset.__getitem__(200)

# print(img.shape, imPts.shape, lidar_fov.shape, gt_img.shape)

def get_datasets(imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, split=False, val_size=0.20):

    kitti_dataset = kittiDataset(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
                                 data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam, transforms=get_transform)
    if split:
        if val_size >= 1:
            raise AssertionError(
                "val_size must be a value within the range of (0,1)")

        len_val = math.ceil(len(kitti_dataset)*val_size)
        len_train = len(kitti_dataset) - len_val

        if len_train < 1 or len_val < 1:
            raise AssertionError("datasets length cannot be zero")
        train_set, val_set = torch.utils.data.random_split(
            kitti_dataset, [len_train, len_val])
        return train_set, val_set
    else:
        return kitti_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(batch_size, imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, split=False, val_size=0.20):

    if split:
        train_set, val_set = get_datasets(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
                                          data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam,
                                          split=split, val_size=val_size)

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
        dataset = get_datasets(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
                               data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam)

        

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=collate_fn,
                                                  drop_last=True)
        return data_loader


imgs_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/val/")
data_depth_velodyne_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_velodyne/val/")
data_depth_annotated_root = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_annotated/val/")

calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_velo_to_cam.txt")
calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_cam_to_cam.txt")

kitti_data_loader = get_dataloaders(batch_size=1, imgs_root=imgs_root,
                                    data_depth_velodyne_root=data_depth_velodyne_root, data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam)


# for img, imPts, lidar_fov, gt_img in kitti_data_loader:
#     print(img.shape, imPts.shape, lidar_fov.shape, gt_img.shape)
iterator = iter(kitti_data_loader)

img, imPts, lidar_fov, gt_img = next(iterator)

print(img[0].shape, imPts[0].shape, lidar_fov[0].shape, gt_img[0].shape)