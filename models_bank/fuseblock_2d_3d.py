
import torch
from torch import nn
import torch.nn.functional as F
from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.continuous_conv import ContinuousConvolution
from utils.lidar_cam_projection import *
from utils.tensorize_batch import tensorize_batch
import config
import temp_variables
import matplotlib.pyplot as plt
import numpy as np
import os


class Two_D_Branch(nn.Module):
    def __init__(self, backbone_out_channels):
        super(Two_D_Branch, self).__init__()

        self.conv1 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv2 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

        self.conv3 = nn.Sequential(
            depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1),
            nn.BatchNorm2d(backbone_out_channels)
        )

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, features, anns=None):

        original_shape = features.shape[2:]

        conv1_out = F.relu(self.conv1(features))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = F.interpolate(conv2_out, original_shape)
        conv3_out = F.relu(self.conv3(features))

        return conv2_out + conv3_out


class Three_D_Branch(nn.Module):
    def __init__(self, backbone_out_channels):
        super(Three_D_Branch, self).__init__()

    def to(self, device):
        for module in self.children():
            module.to(device)

    def show_points(self, imgs, lidar_points_fov, pts_2d_fov, proj_lidar2cam):

        batch_size = len(pts_2d_fov)

        for i in range(batch_size):
            img = imgs[i]
            lidar_points = lidar_points_fov[i]
            # pts_2d = N x 2
            pts_2d = pts_2d_fov[i]
            num_points = pts_2d.shape[0]

            # Homogeneous coords
            ones = torch.ones(
                (lidar_points.shape[0], 1), device=temp_variables.DEVICE)
            lidar_points = torch.cat([lidar_points, ones], dim=1)

            # lidar_points_2_cam = 3 x N
            lidar_points_2_cam = torch.matmul(
                proj_lidar2cam, lidar_points.transpose(0, 1))
            # show lidar points on image
            cmap = plt.cm.get_cmap('hsv', 256)
            cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

            for idx in range(num_points):
                depth = lidar_points_2_cam[2, idx]
                color = cmap[int(640.0 / depth), :]
                cv2.circle(img, (int(torch.round(pts_2d[idx, 0])),
                                 int(torch.round(pts_2d[idx, 1]))),
                           2, color=tuple(color), thickness=-1)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            plt.show()

    def find_k_nearest(self, k, batch_lidar_fov):
        

        distances = torch.cdist(batch_lidar_fov, batch_lidar_fov, p=2)
        _, indices = torch.topk(distances, k + 1, dim=2, largest=False)
        indices = indices[:, :, 1:] # B x N x 3
        return indices


    def forward(self, features, lidar_points, proj_lidar2cam, imgs):
        """
            features = [batch_size, C, h, w]
            lidar_points = tuple(batch_size, npoints, 3)
            proj_lidar2cam = [3, 4]
        """
        # batch_size
        B = features.shape[0]
        C = features.shape[1]
        N = lidar_points.shape[1]
        # height and width
        h, w = features.shape[2:]

        # project lidar to image
        pts_2d = project_to_image_torch(lidar_points.transpose(1, 2), proj_lidar2cam)
        pts_2d = pts_2d.transpose(1,2)

        batch_pts_fov = []
        batch_lidar_fov = []
        batch_f = []

        # # print(lidar_points[0].shape)
        for idx, points in enumerate(pts_2d):
            # find points within image range and in front of lidar
            inds = torch.where((points[:, 0] < w -1) & (points[:, 0] >= 0) &
                                (points[:, 1] < h -1) & (points[:, 1] >= 0) &
                                (lidar_points[idx][:, 0] > 0))
            batch_lidar_fov.append(lidar_points[idx][inds])
            batch_pts_fov.append(points[inds])
            
            pts = points[inds]
            num_points = points[inds].shape[0]
            f = torch.zeros((num_points, C))
            # Get features at projected points
            for i in range(num_points):
                x, y = pts[i]
                x, y = int(torch.round(x)), int(torch.round(y))
                f[i,:] =  features[idx,:,y,x]
            batch_f.append(f)

        batch_pts_fov = tensorize_batch(batch_pts_fov, temp_variables.DEVICE)
        batch_lidar_fov = tensorize_batch(batch_lidar_fov, temp_variables.DEVICE)
        batch_f = tensorize_batch(batch_f, temp_variables.DEVICE)
        
        batch_k_nn_indices = self.find_k_nearest(3, batch_lidar_fov)

        return batch_f, batch_lidar_fov, batch_k_nn_indices
        # return batch_pts_fov, batch_f, batch_k_nn_indices


class FuseBlock(nn.Module):
    def __init__(self,
                 backbone_out_channels):
        super(FuseBlock, self).__init__()

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, images, anns=None, semantic=True, instance=True):
        return 0
