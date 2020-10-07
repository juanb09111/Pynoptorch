
import torch
from torch import nn
import torch.nn.functional as F
from common_blocks.depth_wise_conv import depth_wise_conv
from utils.lidar_cam_projection import *
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

    def show_points(self, img, lidar_points_fov, pts_2d_fov, proj_lidar2cam):

        batch_size = len(pts_2d_fov)

        for i in range(batch_size):
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

            for i in range(num_points):
                depth = lidar_points_2_cam[2, i]
                color = cmap[int(640.0 / depth), :]
                cv2.circle(img, (int(torch.round(pts_2d[i, 0])),
                                 int(torch.round(pts_2d[i, 1]))),
                           2, color=tuple(color), thickness=-1)
            plt.imshow(img)
            plt.yticks([])
            plt.xticks([])
            plt.show()

    def forward(self, features, lidar_points, proj_lidar2cam, img):
        """
            features = [batch_size, C, h, w]
            lidar_points = [batch_size, npoints, 3]
            proj_lidar2cam = [3, 4]
        """
        # height and width
        h, w = features.shape[2:]

        # project lidar to image
        pts_2d = project_to_image_torch(
            lidar_points.transpose(1, 2), proj_lidar2cam)
        pts_2d = pts_2d.transpose(1, 2)

        batch_pts_fov = []
        batch_lidar_fov = []

        for idx, points in enumerate(pts_2d):
            # find points within image range and in front of lidar
            inds = torch.where((points[:, 0] < w) & (points[:, 0] >= 0) &
                               (points[:, 1] < h) & (points[:, 1] >= 0) &
                               (lidar_points[idx,:,0] > 0))
            batch_lidar_fov.append(lidar_points[idx][inds])
            batch_pts_fov.append(points[inds])

        self.show_points(img, batch_lidar_fov, batch_pts_fov, proj_lidar2cam)

        return batch_pts_fov


class FuseBlock(nn.Module):
    def __init__(self,
                 backbone_out_channels):
        super(FuseBlock, self).__init__()

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, images, anns=None, semantic=True, instance=True):
        return 0


# features = torch.rand(2, 256, 512, 512)

# model = Two_D_Branch(256)

# model.eval()

# out = model(features)

# print(out.shape)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", device)
temp_variables.DEVICE = device


# lidar_points = torch.randint(1, 1024, (2, 200, 3), dtype=torch.float, device=device)
# proj_lidar2cam = torch.randint(1, 1000, (3,4), device=device, dtype=torch.float)
# # print(lidar_points, proj_lidar2cam)

model = Three_D_Branch(256)
model.to(device)
model.eval()

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", config.LIDAR_DATA)

# Load image, calibration file, label bbox
png_file = os.path.join(data_folder, "000114_image.png")

rgb = cv2.cvtColor(cv2.imread(png_file), cv2.COLOR_BGR2RGB)
img_height, img_width, img_channel = rgb.shape
features = torch.rand((2, 256, img_height, img_width), device=device)
# Load calibration
calib_file = os.path.join(data_folder, "000114_calib.txt")
calib = read_calib_file(calib_file)
proj_velo2cam2 = project_velo_to_cam2(calib)
proj_velo2cam2 = torch.tensor(
    proj_velo2cam2, device=temp_variables.DEVICE, dtype=torch.float)


# Load Lidar PC
pc_velo_file = os.path.join(data_folder, "000114.bin")
pc_velo = load_velo_scan(pc_velo_file)[:, :3]
pc_velo = torch.tensor(
    [pc_velo, pc_velo], device=temp_variables.DEVICE, dtype=torch.float)



out = model(features, pc_velo, proj_velo2cam2, rgb)

# print(out)
