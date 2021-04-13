# %%
import scipy.io
import os.path
import math
import torch
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import config_kitti
from utils.lidar_cam_projection import read_calib_file, load_velo_scan, full_project_velo_to_cam2, project_to_image
import glob
import cv2
import matplotlib.pyplot as plt
from utils.fill_depth_colorization import fill_depth_colorization
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


def get_file_lists(imgs_root, data_depth_velodyne_root, data_depth_annotated_root, n_samples=None):

    imgs, sync_folders = getListOfImgs(imgs_root)
    (data_velo, data_ann), source_imgs, lidar_file_list = get_depth_data(
        imgs_root, data_depth_velodyne_root, data_depth_annotated_root, sync_folders, imgs)

    if n_samples is None:
        return np.sort(source_imgs), np.sort(data_velo), np.sort(data_ann), np.sort(lidar_file_list)
    
    else:
        return np.sort(source_imgs)[:n_samples], np.sort(data_velo)[:n_samples], np.sort(data_ann)[:n_samples], np.sort(lidar_file_list)[:n_samples]


class kittiDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, transforms=None, n_samples=None):

        self.imgs_root = imgs_root
        self.data_depth_velodyne_root = data_depth_velodyne_root
        self.data_depth_annotated_root = data_depth_annotated_root
        self.calib_velo2cam = calib_velo2cam
        self.calib_cam2cam = calib_cam2cam

        source_imgs, data_velo_files, data_ann_files, lidar_files = get_file_lists(
            imgs_root, data_depth_velodyne_root, data_depth_annotated_root, n_samples = n_samples)

        print("Training on {} samples".format(len(source_imgs)))
        self.source_imgs = source_imgs
        self.data_velo_files = data_velo_files
        self.data_ann_files = data_ann_files
        self.lidar_files = lidar_files
        self.transforms = transforms

    def visualize_projection(self, img_filename, imgfov_pc_pixel, imgfov_pc_cam2, upper, lower, left, right):
        img = cv2.imread(img_filename)[int(
            upper):int(lower), int(left):int(right)]
        crop_img = img[int(upper):int(lower), int(left):int(right)]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    def pre_process_points(self, imPts, lidar_fov, imgfov_pc_cam2):

        _, indices = torch.unique(imPts, dim=0, return_inverse=True)
        unique_indices = torch.zeros_like(torch.unique(indices))

        current_pos = 0
        for i, val in enumerate(indices):
            if val not in indices[:i]:
                unique_indices[current_pos] = i
                current_pos += 1

        imPts = imPts[unique_indices]
        lidar_fov = lidar_fov[unique_indices]
        imgfov_pc_cam2 = imgfov_pc_cam2[:, unique_indices]

        # rand_indices = torch.randint(0, imPts.shape[0], (config_kitti.N_NUMBER, 1))
        rand_perm = torch.randperm(imPts.shape[0])
        imPts = imPts[rand_perm, :]
        lidar_fov = lidar_fov[rand_perm, :]
        imgfov_pc_cam2 = imgfov_pc_cam2[:, rand_perm]
        # print("randint", rand_perm.shape, imPts.shape, lidar_fov.shape)
        return imPts[:config_kitti.N_NUMBER, :], lidar_fov[:config_kitti.N_NUMBER, :], imgfov_pc_cam2[:, :config_kitti.N_NUMBER]

    def find_k_nearest(self, lidar_fov):
        k_number = config_kitti.K_NUMBER
        b_lidar_fov = torch.unsqueeze(lidar_fov, dim=0)

        distances = torch.cdist(b_lidar_fov, b_lidar_fov, p=2)
        _, indices = torch.topk(distances, k_number + 1, dim=2, largest=False)
        indices = indices[:, :, 1:]  # B x N x 3

        return indices.squeeze_(0).long()

    def __getitem__(self, index):

        img_filename = self.source_imgs[index]  # rgb image
        img_name = img_filename.split("/")[-1].split(".")[0]
        lidar_filename = self.lidar_files[index]  # lidar pointcloud

        # velo_2d_filename = self.data_velo_files[index] # 2d lidar points, sparse depth
        sparse_depth_gt = self.data_ann_files[index]  # semi dense depth map
        # print(sparse_depth_gt, img_filename, lidar_filename)

        if img_filename.split("/")[-1].split(".")[0] != sparse_depth_gt.split("/")[-1].split(".")[0] or img_filename.split("/")[-1].split(".")[0] != lidar_filename.split("/")[-1].split(".")[0]:
            raise ValueError("file mismatch ")

        calib_velo2cam = read_calib_file(self.calib_velo2cam)  # projection mat
        calib_cam2cam = read_calib_file(self.calib_cam2cam)  # projection mat

        img = Image.open(img_filename)
        sparse_depth_gt = Image.open(sparse_depth_gt)
        # velo_2d = Image.open(velo_2d_filename)
        # img width and height
        (img_width, img_height) = (img.width, img.height)

        if self.transforms is not None:
            img = self.transforms(crop=True)(img)
            sparse_depth_gt = self.transforms(crop=True)(sparse_depth_gt)

        # ------------------------
        # input_image = cv2.imread(img_filename, cv2.IMREAD_COLOR)  # uint8 image
        # input_image = cv2.normalize(input_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        sparse_depth_gt = np.array(
            sparse_depth_gt, dtype=int).astype(np.float)/256

        # # # print("rgb", norm_image)
        # # # print("gt_input", np.max(gt_input), np.min(gt_input))
        # dense_depth_gt = fill_depth_colorization(input_image, velo_2d)

        # dense_depth_gt = torch.from_numpy(dense_depth_gt)
        # -------------------
        # sparse_depth_gt = np.array(sparse_depth_gt, dtype=int).astype(np.float)/256
        sparse_depth_gt = torch.from_numpy(sparse_depth_gt)
        # -----------------------------

        pc_velo = load_velo_scan(lidar_filename)[:, :3]
        # print("pc_velo", pc_velo.shape)

        # projection matrix (project from velo2cam2)
        proj_velo2cam2 = full_project_velo_to_cam2(
            calib_velo2cam, calib_cam2cam)

        # apply projection
        pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

        # crop boundaries
        upper = np.ceil((img_height - config_kitti.CROP_OUTPUT_SIZE[0])/2)
        lower = np.floor(
            (img_height - config_kitti.CROP_OUTPUT_SIZE[0])/2) + config_kitti.CROP_OUTPUT_SIZE[0]

        left = np.ceil((img_width - config_kitti.CROP_OUTPUT_SIZE[1])/2)
        right = np.floor(
            (img_width - config_kitti.CROP_OUTPUT_SIZE[1])/2) + config_kitti.CROP_OUTPUT_SIZE[1]

        inds = np.where((pts_2d[0, :] < right - 1) & (pts_2d[0, :] > left + 1) & (pts_2d[0, :] >= 0) &
                        (pts_2d[1, :] < lower - 1) & (pts_2d[1, :] > upper + 1) & (pts_2d[1, :] >= 0) &
                        (pc_velo[:, 0] > 0)
                        )[0]

        # move to origin
        pts_2d[0, :] = pts_2d[0, :] - left
        pts_2d[1, :] = pts_2d[1, :] - upper

        # Filter out pixels points
        imgfov_pc_pixel = pts_2d[:, inds]  # pointcloud2pixels

        # Retrieve depth from lidar
        imgfov_pc_velo = pc_velo[inds, :]  # Lidar points within field of view
        imgfov_pc_velo_homo = np.hstack((imgfov_pc_velo, np.ones(
            (imgfov_pc_velo.shape[0], 1))))  # homogeneous coords
        # Lidar projected to cam2
        imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo_homo.transpose()

        # print("imgfov_pc_velo", imgfov_pc_velo[:, 0].max(), imgfov_pc_velo[:, 0].min())
        # fig = plt.figure(4)
        # # ax = plt.axes(projection="3d")
        # ax = plt.axes(projection='3d')

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # x_data = imgfov_pc_velo[:, 0]
        # y_data = imgfov_pc_velo[:, 1]
        # z_data = imgfov_pc_velo[:, 2]
        # ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)
        # plt.show()

        # visualize
        # self.visualize_projection(
        #     img_filename, imgfov_pc_pixel, imgfov_pc_cam2, upper, lower, left, right)

        imPts = torch.tensor(
            imgfov_pc_pixel, dtype=torch.long).permute(1, 0)  # N x 2

        lidar_fov = torch.tensor(
            imgfov_pc_velo, dtype=torch.float)  # N x 3

        # print(lidar_fov.shape, imgfov_pc_velo_homo.shape)
        # remove duplicate
        # print(lidar_fov.shape)
        imPts, lidar_fov, imgfov_pc_cam2 = self.pre_process_points(
            imPts, lidar_fov, imgfov_pc_cam2)

        # print(imPts.shape)

        # if self.transforms is not None:
        #     img = self.transforms(resize=True)(img)

        mask = torch.zeros(img.shape[1:], dtype=torch.bool)
        # # print(mask.shape)
        # # mask = torch.zeros_like(img[0,:,:].squeeze_(0), dtype=torch.bool)
        sparse_depth = torch.zeros_like(
            img[0, :, :].unsqueeze_(0), dtype=torch.float)

        mask[imPts[:, 1], imPts[:, 0]] = True
        sparse_depth[0, imPts[:, 1], imPts[:, 0]] = torch.tensor(
            imgfov_pc_cam2[2, :], dtype=torch.float)

        # print(lidar_fov[:, 0].max(), lidar_fov[:, 0].min())

        # plt.imshow(mask)
        # print('mask')
        # plt.show()

        # plt.imshow(sparse_depth.permute(1,2,0))
        # print('sparse_depth', sparse_depth.shape, sparse_depth.max(), sparse_depth.min())
        # plt.show()

        # plt.imshow(sparse_depth_gt.permute(1,2,0))
        # print('sparse_depth_gt', sparse_depth_gt.shape, sparse_depth_gt.max(), sparse_depth_gt.min())
        # plt.show()
        k_nn_indices = self.find_k_nearest(lidar_fov)
        # print(img.shape, lidar_fov.shape, mask.shape, sparse_depth.shape, k_nn_indices.shape, sparse_depth_gt.shape)
        # print("max", torch.max(gt_img), torch.min(gt_img), gt_img.shape)
        # print(img.shape, imPts.shape, lidar_fov.shape, mask.shape, sparse_depth.shape, k_nn_indices.shape, gt_img.shape)
        # return img, imPts, lidar_fov, mask, sparse_depth, k_nn_indices, gt_img

        # img: source image
        # lidar_fov: lidar points within the field of view of the camera
        # mask: boolean mask where the points are projected
        # sparse_depth: lidar projected on to cam2
        # k_nn_indices: indices of the knn
        # sparse_depth_gt: sparse depth ground truth
        return img, lidar_fov, mask, sparse_depth, k_nn_indices, sparse_depth_gt, img_name

    def __len__(self):
        return len(self.source_imgs)



def get_transform(resize=False, normalize=False, crop=False):
    new_size = tuple(np.ceil(x*config_kitti.RESIZE)
                     for x in config_kitti.ORIGINAL_INPUT_SIZE_HW)
    new_size = tuple(int(x) for x in new_size)
    custom_transforms = []
    if resize:
        custom_transforms.append(transforms.Resize(new_size))

    if crop:
        custom_transforms.append(
            transforms.CenterCrop(config_kitti.CROP_OUTPUT_SIZE))

    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Lambda(lambda image: torch.from_numpy(np.array(image).astype(np.float32)/255).unsqueeze(0)))
    if normalize:
        custom_transforms.append(transforms.Normalize(0.485, 0.229))
    return transforms.Compose(custom_transforms)


# imgs_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/train/")
# data_depth_velodyne_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "data_depth_velodyne/train/")
# data_depth_annotated_root_train = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "data_depth_annotated/train/")

# calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/calib_velo_to_cam.txt")
# calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "..", config_kitti.DATA, "imgs/2011_09_26/calib_cam_to_cam.txt")

# kitti_dataset = kittiDataset(imgs_root_train, data_depth_velodyne_root_train,
#                              data_depth_annotated_root_train, calib_velo2cam, calib_cam2cam, transforms=get_transform, n_samples=600)
# for i in range(600):

#     # kitti_dataset.__getitem__(np.random.randint(500))
#     kitti_dataset.__getitem__(i)

def get_datasets(imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, split=False, val_size=0.20, n_samples=None):

    kitti_dataset = kittiDataset(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
                                 data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam, transforms=get_transform, n_samples=n_samples)
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


def get_dataloaders(batch_size, imgs_root, data_depth_velodyne_root, data_depth_annotated_root, calib_velo2cam, calib_cam2cam, split=False, val_size=0.20, n_samples=None):

    # if split:
    #     train_set, val_set = get_datasets(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
    #                                       data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam,
    #                                       split=split, val_size=val_size)

    #     data_loader_train = torch.utils.data.DataLoader(train_set,
    #                                                     batch_size=batch_size,
    #                                                     shuffle=False,
    #                                                     num_workers=0,
    #                                                     collate_fn=collate_fn,
    #                                                     drop_last=True)

    #     data_loader_val = torch.utils.data.DataLoader(val_set,
    #                                                   batch_size=batch_size,
    #                                                   shuffle=False,
    #                                                   num_workers=0,
    #                                                   collate_fn=collate_fn,
    #                                                   drop_last=True)
    #     return data_loader_train, data_loader_val

    # else:
    dataset = get_datasets(imgs_root=imgs_root, data_depth_velodyne_root=data_depth_velodyne_root,
                            data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam, n_samples=n_samples)

    data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,
                                                collate_fn=collate_fn,
                                                drop_last=True)
    return data_loader


# imgs_root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/train/")
# data_depth_velodyne_root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_velodyne/train/")
# data_depth_annotated_root = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/data_depth_annotated/train/")

# calib_velo2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_velo_to_cam.txt")
# calib_cam2cam = calib_filename = os.path.join(os.path.dirname(os.path.abspath(
#     __file__)), "../data_kitti/kitti_depth_completion_unmodified/imgs/2011_09_26/calib_cam_to_cam.txt")

# kitti_data_loader = get_dataloaders(batch_size=1, imgs_root=imgs_root,
#                                     data_depth_velodyne_root=data_depth_velodyne_root, data_depth_annotated_root=data_depth_annotated_root, calib_velo2cam=calib_velo2cam, calib_cam2cam=calib_cam2cam)


# iterator = iter(kitti_data_loader)

# # (img, imPts, lidar_fov, mask, sparse_depth), gt_img = next(iterator)

# img, imPts, lidar_fov, mask, sparse_depth, k_nn_indices, gt_img = next(iterator)

# # print(img[0].shape, imPts[0].shape, lidar_fov[0].shape, gt_img[0].shape)

# # img, imPts, lidar_fov, mask, sparse_depth = data_tuple[0]


# for inputs in kitti_data_loader:

#     img, imPts, lidar_fov, mask, sparse_depth, k_nn_indices, gt_img = inputs


#     print(img[0].shape, imPts[0].shape, lidar_fov[0].shape, mask[0].shape, sparse_depth[0].shape, k_nn_indices[0].shape, gt_img[0].shape)
