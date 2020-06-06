import torch
from torch import nn
import torch.nn.functional as F
import torchvision
#%%
class box_head(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, representation_size, bias= True),
            nn.BatchNorm1d(representation_size)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(representation_size, representation_size, bias= True),
            nn.BatchNorm1d(representation_size)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x


# x= torch.rand(1024, 256, 14, 14)

# model = box_head(256*14 ** 2, 1024)

# out = model(x)

# print(out.shape)
