import torch
import cv2
import config
import models
from utils.tensorize_batch import tensorize_batch
import torchvision.transforms as transforms
from utils.show_bbox import colors_pallete, apply_semantic_mask
import matplotlib.pyplot as plt
from PIL import Image

cap = cv2.VideoCapture(0)


torch.cuda.empty_cache()

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


model = models.get_model()
model.load_state_dict(torch.load(config.MODEL_WEIGHTS_FILENAME))
# move model to the right device
model.to(device)

model.eval()

# model.half()

def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    # custom_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(custom_transforms)


transforms = get_transform()


def view_masks(frame,
               num_classes,
               result_type,
               confidence=0.5):

    image = Image.fromarray(frame.astype('uint8'), 'RGB')
    image = transforms(image)
    images = tensorize_batch([image], device)
    output = model(images)[0]
    if result_type == "semantic":

        logits = output["semantic_logits"]
        mask = torch.argmax(logits, dim=0)
        mask = mask.cpu().numpy()
        im = apply_semantic_mask(frame/255, mask, colors_pallete)
        return im


while(True):
    ret, frame = cap.read()

    if config.RT_SEMANTIC:
        im = view_masks(frame, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                        "semantic",
                        confidence=0.5)
        cv2.imshow('frame', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
