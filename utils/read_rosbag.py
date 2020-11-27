import scipy.io
import os.path
import math
import torch
import numpy as np

from utils.tensorize_batch import tensorize_batch
from utils.lidar_cam_projection import *

import matplotlib.pyplot as plt

import config
import cv2


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", device)

temp_variables.DEVICE = device

# Empty cuda cache
torch.cuda.empty_cache()

polar_point_cloud_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "lidar_data/JD/polar_point_cloud.mat")


lidar_fov_file = polar_point_cloud_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "lidar_data_jd/lidar_fov.mat")

mat2 = scipy.io.loadmat(lidar_fov_file)["lidar_fov"]
    
mat = scipy.io.loadmat(polar_point_cloud_file)

polar_point_cloud = mat["polar_pointcloud"]

alkukulma = math.radians(1)
loppukulma = math.radians(359)

SeqId = 3

n_samples = polar_point_cloud.size

samples_seqId = polar_point_cloud[:, 96] == 3

# create tensors

polar_point_cloud_tensor = torch.tensor(polar_point_cloud, device=device)


inds = torch.where((polar_point_cloud_tensor[:, 96] == SeqId) & (
    polar_point_cloud_tensor[:, 2] >= alkukulma) & (polar_point_cloud_tensor[:, 2] <= loppukulma))


polar_point_cloud_tensor = polar_point_cloud_tensor[inds]
n_samples = polar_point_cloud_tensor.shape[0]

n_lidar_rows = 16
coords = torch.zeros((n_samples, 16, 3), device=device)

range_indices = [1, 13, 25, 37, 49, 61, 73, 85, 7, 19, 31, 43, 55, 67, 79, 91]
rot_angles = [x + 1 for x in range_indices]
vert_angles = [x + 2 for x in range_indices]
# samples_seqId_tensor = torch.tensor(samples_seqId, device=device

coords[:, 0:n_lidar_rows, 0] = polar_point_cloud_tensor[:, range_indices]
coords[:, 0:n_lidar_rows, 1] = polar_point_cloud_tensor[:, rot_angles]
coords[:, 0:n_lidar_rows, 2] = polar_point_cloud_tensor[:, vert_angles]

coords = coords.view(-1, 3)
cart_coords = torch.zeros_like(coords, device=device)

# convert to  xyz coord
cart_coords[:, 0] = coords[:, 0]*torch.cos(coords[:, 1])
cart_coords[:, 1] = -coords[:, 0]*torch.sin(coords[:, 1])
cart_coords[:, 2] = torch.tan(
    coords[:, 2])*torch.sqrt(torch.square(cart_coords[:, 0])+torch.square(cart_coords[:, 1]))

# homogenous coords
n_points = cart_coords.shape[0]
new_axis = torch.ones((n_points, 1), device=device)
cart_coords = torch.cat([cart_coords, new_axis], 1)

# Transformation

rotX = torch.tensor([0.0349], device=device)
rotY = torch.tensor([-0.1222], device=device)
rotZ = torch.tensor([-1.3491], device=device)


rotY_m = torch.tensor([[torch.cos(rotY), 0, torch.sin(rotY)], [
                      0, 1, 0], [-torch.sin(rotY), 0, torch.cos(rotY)]], device=device)

rotZ_m = torch.tensor([[torch.cos(rotZ), -torch.sin(rotZ), 0], [torch.sin(
    rotZ), torch.cos(rotZ), 0], [0, 0, 1]], device=device)

rotX_m = torch.tensor([[1, 0, 0], [0, torch.cos(
    rotX), -torch.sin(rotX)], [0, torch.sin(rotX), torch.cos(rotX)]], device=device)

trans = torch.tensor([[0, 0.33, 0.2]], device=device).transpose_(0, 1)

# compose rigid transformation
hom_row = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float)

rot_trans = torch.cat((rotZ_m, trans), 1)
T1 = torch.cat((rot_trans, hom_row), 0)  # rotz and translation

T2 = torch.cat((rotY_m, torch.tensor(
    [[0], [0], [0]], device=device, dtype=torch.float)), 1)
T2 = torch.cat((T2, hom_row), 0)  # roty

T3 = torch.cat((rotX_m, torch.tensor(
    [[0], [0], [0]], device=device, dtype=torch.float)), 1)
T3 = torch.cat((T3, hom_row), 0)  # rotx

T4 = torch.tensor([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                  dtype=torch.float, device=device)  # axes swap lidar coord --> cam coord


T_total = torch.matmul(torch.matmul(T4, T3), torch.matmul(T2, T1))
# print("t_total", T_total) 
# Apply transformation

cart_coords = cart_coords.transpose_(0, 1) # 3 x N
cart_coords = torch.matmul(T_total, cart_coords) # 3 x N

print("c", cart_coords[:, 889:895])
inds = torch.where((cart_coords[2,:] > 0))

inds_z = torch.where((cart_coords[2,:] <= -0.0244))
print("inds_z", inds_z)

print("cart_cords shape -before", cart_coords.shape, len(inds))
cart_coords = cart_coords.permute((1, 0))[inds]
print("cart_cords shape -after", cart_coords.shape)

#%% show transform
# fig = plt.figure(4)
# # ax = plt.axes(projection="3d")
# ax = plt.axes(projection='3d')

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# x_data = cart_coords[:, 0].cpu().numpy()
# y_data = cart_coords[:, 1].cpu().numpy()
# z_data = cart_coords[:, 2].cpu().numpy()

# # ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', s=1)
# ax.scatter3D(x_data, y_data, z_data, cmap='Greens', s=1)

# plt.show()
#%%
# #conver to meters
# # cart_coords = cart_coords/1000

# intrisic_m = torch.tensor([[0.9621325, 0, 0], [0, 0.9678951, 0], [0.6377971, 0.4418724, 1]], device=device)

# intrisic_m = intrisic_m.transpose_(0, 1)


# projected_points = torch.matmul(intrisic_m, cart_coords.transpose_(0,1)[0:3])
# print("proj shape", projected_points.shape)
# projected_points[:2, :] /= projected_points[2, :]

# print("proj shape", projected_points.shape)
# # print("projected_points", projected_points[0:2, 953:1000])

# fig = plt.figure(3)
# ax = plt.axes(projection="3d")
# # ax = plt.axes()

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# x_data = projected_points[0, :].cpu().numpy()
# y_data = projected_points[1, :].cpu().numpy()
# z_data = projected_points[2, :].cpu().numpy()

# ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', s=1)
# # ax.scatter(x_data, y_data, c=z_data, cmap='Greens', s=1)

# plt.show()


#%% Network --------------------------------------------------------------

# # lidar img
# lidar_img_filename=  os.path.join(os.path.dirname(
#             os.path.abspath(__file__)), "..", config.LIDAR_DATA, "img.png")

# img = cv2.cvtColor(cv2.imread(lidar_img_filename), cv2.COLOR_BGR2RGB)

# lidar_imgs = [img, img]
# img_height, img_width, img_channel = lidar_imgs[0].shape

# # # features
# features = torch.rand((2, 256, img_height, img_width), device=device)
# C = features.shape[1]

# # # # calib m
# calib = [intrisic_m, intrisic_m]
# calib = [cal.to(device) for cal in calib]
# calib = calib[0]

# # #lidar data
# lidar_data = [cart_coords[:, :3], cart_coords[:, :3]]
# lidar_data = tensorize_batch(lidar_data, device)
# print("1", lidar_data)
# # k_number
# k_number = 3

# # # # pre-process
# mask, batch_lidar_fov, batch_pts_fov, batch_k_nn_indices = pre_process_points(features, lidar_data, calib, k_number)

# plt.imshow(mask[0].cpu().numpy())
# plt.show()

# show_lidar_2d(lidar_imgs, batch_lidar_fov, batch_pts_fov, calib)
