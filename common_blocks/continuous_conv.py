
import torch
import torch.nn as nn
import temp_variables
class ContinuousConvolution(nn.Module):
    """
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Deep_Parametric_Continuous_CVPR_2018_paper.pdf
    h_i = W(Sum_k MLP(x_i - x_k) * f_k)
    inputs:
    + x: B x N x C (points features)
    + points: B x N x 3 (points coordinates)
    + indices: B x N x K (knn indices)

    outputs:
    + y: points features
    """

    def __init__(self, n_feat, k_number, n_number):
        super().__init__()

        self.mlp = nn.Sequential(
            # input: B x N x 3 x K
            nn.Linear(3 * k_number, (n_feat // 2) * k_number),  # B x N x 3*(n_feat//2)
            nn.BatchNorm1d(n_number),  # B x N(normalize this dim) x 3*(n_feat//2)
            nn.ReLU(),
            nn.Linear((n_feat // 2) * k_number, n_feat * k_number),  # B x N x n_feat*k_number
            nn.BatchNorm1d(n_number),  # B x N(again, norm this dim) x n_feat*k_number
            nn.ReLU()
        )

    def forward(self, *inputs):
        """
        NOTE: *inputs indicates the input parameter expect a tuple
             **inputs indicates the input parameter expect a dict
        x: B x N x C (points features)
        points: B x N x 3 (points coordinates)
        indices: B x N x K (knn indices)
        """
        x, points, indices = inputs[0]
        B, N, C = x.size()
        K = indices.size(2)

        # print(x.shape, points.shape, indices.shape, B, N, C, K)

        # y1 = torch.zeros((B,N,K,3), device=temp_variables.DEVICE) # B x N x K x 3

        # y2 = torch.zeros((B,N,K,C), device=temp_variables.DEVICE) # B x N x K x C

        # for i in range(B):
        #     idxs = indices[i] # N x K
        #     pts = points[i, idxs] # N x K x 3
        #     y1[i, :, :, :] = pts

        #     feats = x[i, idxs] # N x 3 x C
        #     y2[i, :, :, :] = feats

        y1 = torch.cat([points[i, indices[i]] for i in range(B)], dim=0).view(B,N,K,3)  # B x N x K x 3
        # print("forward3", torch.cuda.memory_allocated(device=temp_variables.DEVICE)/1e9)
        # print(y1.shape, points.shape, points[:, :, None, :].shape)
        # print("y3", y3.shape)
        y1 = points[:, :, None, :] - y1  # B x N x K x 3
        y1 = y1.view(B, N, K * 3)  # B x N x K*3

        # output mlp: B x N x K*C
        y1 = self.mlp(y1).view(B, N, K, C)  # reshape after mlp B x N x K x C

        y2 = torch.cat([x[i, indices[i]] for i in range(B)], dim=0).view(B, N, K, C)  # B x N x K x C
        # print("y2 ,,", y2.shape)
        return torch.sum(y1 * y2, dim=2), points, indices