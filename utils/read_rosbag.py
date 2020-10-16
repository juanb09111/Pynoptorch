# %%
import scipy.io
import os.path
import math
import torch

import numpy as np
import matplotlib.pyplot as plt


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Device: ", device)


polar_point_cloud_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "lidar_data/JD/polar_point_cloud.mat")


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

n_points = cart_coords.shape[0]
new_axis = torch.ones((n_points, 1), device=device)
cart_coords = torch.cat([cart_coords, new_axis], 1)
# Transformation

rotY = torch.tensor([-0.0611], device=device)
rotZ = torch.tensor([-1.3614], device=device)


rotY_m = torch.tensor([[torch.cos(rotY), 0, torch.sin(rotY), 0], [
                      0, 1, 0, 0], [-torch.sin(rotY), 0, torch.cos(rotY), 0], [0, 0, 0, 1]], device=device)

rotZ_m = torch.tensor([[torch.cos(rotZ), -torch.sin(rotZ), 0, 0], [torch.sin(
    rotZ), torch.cos(rotZ), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device)


# Transformation 1 rotz
cart_coords = torch.matmul(cart_coords, rotZ_m)

# Transformation 2 translation

offset = torch.tensor([0, 0.33, 0.2, 0], device=device)

cart_coords = cart_coords + offset

# Transformation 3 Roty

cart_coords = torch.matmul(cart_coords, rotY_m)
# %%

# plot

fig = plt.figure(2)
ax = plt.axes(projection='3d')

# ax.set_xlim([0, 20])
# ax.set_ylim([-5, 5])
# ax.set_zlim([-5, 4])

# inds = torch.where((cart_coords[:, 0] > 0) & (cart_coords[:, 0] < 20) & (cart_coords[:, 1] > -5) &
#                    (cart_coords[:, 1] < 5) & (cart_coords[:, 2] > -5) & (cart_coords[:, 2] < 4))

# cart_coords = cart_coords[inds]

x_data = cart_coords[:, 0].cpu().numpy()
y_data = cart_coords[:, 1].cpu().numpy()
z_data = cart_coords[:, 2].cpu().numpy()


ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap='Greens', s=1)

plt.show()
# %%
