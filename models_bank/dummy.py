#%%
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DummyNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.transform = model.transform
        self.backbone = model.backbone
        self.classifier = DeepLabHead(model.backbone.out_channels, num_classes)
        # self.hidden = nn.Linear(784, 256)

        # self.output = nn.Linear(256, 10)

        # self.sigmoid = nn.Sigmoid()

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        # x = F.sigmoid(self.hidden(x))

        # # x = F.softmax(self.output(x), dim=1)

        # x = F.log_softmax(self.output(x), dim=1)

        x = self.transform(x)
        x = self.backbone(x)
        x = self.classifier(x)

        return x

def get_model(num_classes):
    return DummyNetwork(num_classes)
# %%
