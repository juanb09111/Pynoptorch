
import torch
from torch import nn
import torch.nn.functional as F
from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.depth_wise_sep_conv import depth_wise_sep_conv
from common_blocks.continuous_conv import ContinuousConvolution
import temp_variables
import config_kitti


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

    def forward(self, features):

        original_shape = features.shape[2:]
        conv1_out = F.relu(self.conv1(features))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv2_out = F.interpolate(conv2_out, original_shape)
        conv3_out = F.relu(self.conv3(features))

        return conv2_out + conv3_out


class Three_D_Branch(nn.Module):
    def __init__(self, n_feat, k_number, n_number):
        super(Three_D_Branch, self).__init__()

        self.branch_3d_continuous = nn.Sequential(
            ContinuousConvolution(n_feat, k_number, n_number),
            ContinuousConvolution(n_feat, k_number, n_number)
        )


    def forward(self, feats, mask, coors, indices):
        """
        mask: B x H x W
        feats: B x C x H x W
        coors: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices, aka. mask_knn)
        """
        
        B, C, _, _ = feats.shape
        feats_mask = feats.permute(0, 2, 3, 1)[mask].view(B, -1, C)
        br_3d, _, _ = self.branch_3d_continuous((feats_mask, coors, indices)) # B x N x C
        br_3d = br_3d.view(-1, C)  # B*N x C
        
        out = torch.zeros_like(feats.permute(0, 2, 3, 1))  # B x H x W x C
        out[mask] = br_3d
        out = out.permute(0, 3, 1, 2)  # B x C x H x W

        return out
        
        


class FuseBlock(nn.Module):
    def __init__(self, nin, nout, k_number, n_number, extra_output_layer=False):
        super(FuseBlock, self).__init__()

        self.extra_output_layer = extra_output_layer
        self.branch_2d = Two_D_Branch(nin)

        self.branch_3d = Three_D_Branch(nin, k_number, n_number)


        self.output_layer = nn.Sequential(
            # depth_wise_conv(backbone_out_channels, kernel_size=3, stride=1, padding=1),
            depth_wise_sep_conv(nin, nout, kernel_size=3, padding=1),
            nn.BatchNorm2d(nout)
        )


    def forward(self, *inputs):

        # mask: B x H x W
        # feats: B x C x H x W
        # coors: B x N x 3 (points coordinates)
        # indices: B x N x K (knn indices, aka. mask_knn)

        feats, mask, coors, k_nn_indices = inputs[0]
        y = self.branch_3d(feats, mask, coors, k_nn_indices) + self.branch_2d(feats)

        y = F.relu(self.output_layer(y))

        if self.extra_output_layer:
            y = y + feats
            return (y, mask, coors, k_nn_indices)

        return (y, mask, coors, k_nn_indices)


class FuseNet(nn.Module):
    def __init__(self, k_number, n_number):
        super().__init__()

        self.sparse_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.rgbd_conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fuse_conv = nn.Sequential(
            FuseBlock(48, 64, k_number, n_number),
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True), 
            FuseBlock(64, 64, k_number, n_number, extra_output_layer=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        


    def forward(self, img, sparse_depth, mask, coors, k_nn_indices, gt_img):
        """
        inputs:
        img: input rgb (B x 3 x H x W)
        sparse_depth: input sparse depth (B x 1 x H x W)
        coors: sparse 3D points (B x 3 x N)
        mask: mask_2d3d (B x H x W)
        indices: mask_knn (B x N x K)

        output:
        depth: completed depth
        """
        
        _, H, W = mask.shape

        # sparse depth branch
        y_sparse = self.sparse_conv(sparse_depth) # B x 16 x H/2 x W/2

        # rgbd branch
        x_concat_d = torch.cat((img, sparse_depth), dim=1)
        y_rgbd = self.rgbd_conv(x_concat_d)  # B x 32 x H/2 x W/2

        y_rgbd_concat_y_sparse = torch.cat((y_rgbd, y_sparse), dim=1)

        y_rgbd_concat_y_sparse = F.interpolate(y_rgbd_concat_y_sparse, (H, W))
        
        fused, _, _, _ = self.fuse_conv((y_rgbd_concat_y_sparse, mask, coors, k_nn_indices))

        out = self.output_layer(fused)

        if self.training:
            out_original_size = F.interpolate(out, config_kitti.ORIGINAL_INPUT_SIZE_HW).squeeze_(1)
            # print("out", torch.max(out_original_size), torch.min(out_original_size))
            # print("gt_img", torch.max(gt_img), torch.min(gt_img))
            l2 = F.mse_loss(out_original_size, gt_img)
            # l1 = F.smooth_l1_loss(out_original_size, gt_img)

            # total_loss = l2 + l1*torch.tensor((config_kitti.LOSS_ALPHA), device=temp_variables.DEVICE)

            losses = {"depth_loss": l2}

            return losses
        
        return self.output_layer(fused)