#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from .LSFE import LSFE 
from .DPC import DPC 
from .MC import MC
#%%
class instance_segmentation_head(nn.Module):
    def __init__(self, in_channels, num_classes, output_resol):
        super().__init__()

        self.output_resol = output_resol

        self.LSFE_P4 = LSFE(in_channels, 128)
        self.LSFE_P8 = LSFE(in_channels, 128)

        self.DPC_P16 = DPC(in_channels, 128)
        self.DPC_P32 = DPC(in_channels, 128)

        self.MC1 = MC(128, 128)
        self.MC2 = MC(128, 128)

        self.out_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, P4, P8, P16, P32):
        
        out_1 = self.DPC_P32(P32)
        out_2 = self.DPC_P16(P16)

        out_3 = self.LSFE_P8(P8)
        out_4 = self.LSFE_P4(P4)

        out_1_upsampledx2 = F.interpolate(out_1, size=out_2.shape[2:], mode='bilinear')

        out_2_1 = out_2 + out_1_upsampledx2
        out_2_2 = self.MC1(out_2_1)
        
        out_2_2_upsampled = F.interpolate(out_2_2, size=out_3.shape[2:])

        out_2_3 = out_3 + out_2_2_upsampled

        out_2_4 = self.MC2(out_2_3)

        out_2_4_upsampled = F.interpolate(out_2_4, size=out_4.shape[2:])

        out_2_5 = out_2_4_upsampled + out_4

        concat_in_1 = F.interpolate(out_1, size=(256,512), mode='bilinear')
        concat_in_2 = F.interpolate(out_2, size=(256,512), mode='bilinear')
        concat_in_3 = F.interpolate(out_2_3, size=(256,512), mode='bilinear')
        concat_in_4 = F.interpolate(out_2_5, size=(256,512), mode='bilinear')

        x = torch.cat((concat_in_1, concat_in_2, concat_in_3, concat_in_4), dim=1)

        x = self.out_conv(x)

        x = F.interpolate(x, size=self.output_resol)
        return x

