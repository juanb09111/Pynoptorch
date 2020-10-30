
import torch
from torch import nn
import torch.nn.functional as F
from common_blocks.depth_wise_conv import depth_wise_conv
from common_blocks.continuous_conv import ContinuousConvolution
import temp_variables



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
    def __init__(self, n_feat, k_number, n_number):
        super(Three_D_Branch, self).__init__()

        self.branch_3d_continuous = nn.Sequential(
            ContinuousConvolution(n_feat, k_number, n_number),
            ContinuousConvolution(n_feat, k_number, n_number)
        )

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, mask, feats, coors, indices):
        """
        # feats: B x N x C (features at coors)
        feats: B x C x H x W
        coors: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices, aka. mask_knn)
        """
        B, C, _, _ = feats.shape
        feats = feats.permute(0, 2, 3, 1)[mask].view(B, -1, C)
        print(feats.shape, coors.shape, indices.shape)
        br_3d, _, _ = self.branch_3d_continuous((feats, coors, indices)) # B x N x C
        # br_3d = br_3d.view(-1, C)  # B*N x C

        # print(br_3d.shape)
        


class FuseBlock(nn.Module):
    def __init__(self,
                 backbone_out_channels):
        super(FuseBlock, self).__init__()

    def to(self, device):
        for module in self.children():
            module.to(device)

    def forward(self, images, anns=None, semantic=True, instance=True):
        return 0
