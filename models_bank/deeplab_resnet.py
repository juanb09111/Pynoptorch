import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def get_model(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = deeplabv3_resnet101(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    deeplab_head_in_channels = model.classifier[0].convs[0][0].in_channels
    fcn_head_in_channels = model.classifier[1].in_channels

    model.classifier = DeepLabHead(deeplab_head_in_channels, num_classes)

    
    return model