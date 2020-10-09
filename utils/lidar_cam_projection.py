import cv2
import mayavi.mlab as mlab
import numpy as np
import torch
import temp_variables
from utils.tensorize_batch import tensorize_batch


class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = data[1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.occlusion = int(data[2])
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.ry = data[14]

    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d


# =========================================================
# Projections
# =========================================================
def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(
        3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_cam2_to_velo(calib):
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    R_ref2rect_inv = np.linalg.inv(R_ref2rect)  # rect2ref_cam

    # inverse rigid transformation
    velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(
        3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_cam_ref2velo = np.linalg.inv(velo2cam_ref)

    proj_mat = R_ref2rect_inv @ P_cam_ref2velo
    return proj_mat


def project_to_image_torch(points, proj_mat):
    """
    Apply the perspective projection torch version
    Args:
        pts_3d:     3D points in camera coordinate [batch, 3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[2]
    batch_size = points.shape[0]
    # Change to homogenous coordinate
    new_axis = torch.ones((batch_size, 1, num_pts),
                          device=temp_variables.DEVICE)
    points = torch.cat([points, new_axis], 1)
    points = torch.matmul(proj_mat, points)
    for batch in points:
        batch[:2, :] /= batch[2, :]

    return points[:, :2, :]


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def project_camera_to_lidar(points, proj_mat):
    """
    Args:
        points:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]

    Returns:
        points in lidar coordinate:     [3, npoints]
    """
    num_pts = points.shape[1]
    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    return points[:3, :]


def map_box_to_image(box, proj_mat):
    """
    Projects 3D bounding box into the image plane.
    Args:
        box (Box3D)
        proj_mat: projection matrix
    """
    # box in camera coordinate
    points_3d = box.in_camera_coordinate()

    # project the 3d bounding box into the image plane
    points_2d = project_to_image(points_3d, proj_mat)

    return points_2d


# =========================================================
# Utils
# =========================================================
def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


# =========================================================
# Drawing tool
# =========================================================
def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1)):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3,8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0,1), box line color
    """
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)
    return fig


def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    qs = qs.astype(np.int32).transpose()
    for k in range(0, 4):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness, cv2.LINE_AA)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness, cv2.LINE_AA)

    return image


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    """
    Add lidar points
    Args:
        pc: point cloud xyz [npoints, 3]
        color:
        fig: fig handler
    Returns:

    """
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor,
                          fgcolor=None, engine=None, size=(1600, 1000))
    if color is None:
        color = pc[:, 2]

    # add points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # # draw origin
    # mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=fig)

    return fig



def find_k_nearest(k, batch_lidar_fov):
        distances = torch.cdist(batch_lidar_fov, batch_lidar_fov, p=2)
        _, indices = torch.topk(distances, k + 1, dim=2, largest=False)
        indices = indices[:, :, 1:] # B x N x 3
        return indices

def pre_process_points(features, lidar_points, proj_lidar2cam):

    # B = features.shape[0]
    C = features.shape[1]
    # N = lidar_points.shape[1]
    # height and width
    h, w = features.shape[2:]

    # project lidar to image
    pts_2d = project_to_image_torch(
        lidar_points.transpose(1, 2), proj_lidar2cam)
    pts_2d = pts_2d.transpose(1, 2)

    batch_pts_fov = []
    batch_lidar_fov = []
    batch_f = []

       # # print(lidar_points[0].shape)
    for idx, points in enumerate(pts_2d):
        # find points within image range and in front of lidar
        inds = torch.where((points[:, 0] < w - 1) & (points[:, 0] >= 0) &
                            (points[:, 1] < h - 1) & (points[:, 1] >= 0) &
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
            f[i, :] = features[idx, :, y, x]
        batch_f.append(f)

    batch_pts_fov = tensorize_batch(batch_pts_fov, temp_variables.DEVICE)
    batch_lidar_fov = tensorize_batch(
        batch_lidar_fov, temp_variables.DEVICE)
    batch_f = tensorize_batch(batch_f, temp_variables.DEVICE)

    batch_k_nn_indices = find_k_nearest(3, batch_lidar_fov)

    return batch_f, batch_lidar_fov, batch_k_nn_indices
