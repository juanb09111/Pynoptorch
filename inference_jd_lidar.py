import config_jd_lidar
import temp_variables
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch
from torch import nn
from utils.get_jd_testset import get_dataloaders

from datetime import datetime

import matplotlib.pyplot as plt


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()


def RMSE(sparse_depth_gt, pred):
    mask_gt = torch.where(sparse_depth_gt > 0, torch.tensor((1), device=temp_variables.DEVICE,
                                                            dtype=torch.float64), torch.tensor((0), device=temp_variables.DEVICE, dtype=torch.float64))
    mask_gt = mask_gt.squeeze_(1)
    sparse_depth_gt = sparse_depth_gt.squeeze_(
        1)  # remove C dimension there's only one
    c = torch.tensor((1000), device=device)
    sparse_depth_gt = sparse_depth_gt*c
    pred = pred*c
    criterion = nn.MSELoss()
    res = torch.sqrt(criterion(sparse_depth_gt, pred*mask_gt))
    return res


def predict(model,
            data_loader_val,
            weights_file,
            folder):

    # Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    # load weights
    print(weights_file)
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)

    model.eval()

    with torch.no_grad():

        for i, (imgs, lidar_fov, masks, sparse_depth, k_nn_indices, img_names) in enumerate(data_loader_val):

            imgs = list(img for img in imgs)
            lidar_fov = list(lid_fov for lid_fov in lidar_fov)
            masks = list(mask for mask in masks)
            sparse_depth = list(sd for sd in sparse_depth)
            k_nn_indices = list(k_nn for k_nn in k_nn_indices)
            img_names = list(image_name for image_name in img_names)

            imgs = tensorize_batch(imgs, device)
            lidar_fov = tensorize_batch(lidar_fov, device, dtype=torch.float)
            masks = tensorize_batch(masks, device, dtype=torch.bool)
            sparse_depth = tensorize_batch(sparse_depth, device)
            k_nn_indices = tensorize_batch(
                k_nn_indices, device, dtype=torch.long)

            _, outputs = model(imgs, sparse_depth, masks,
                               lidar_fov, k_nn_indices, sparse_depth_gt=None)

            for idx in range(outputs.shape[0]):

                # rmse = RMSE(sparse_depth_gt, outputs[idx])

                out = outputs[idx].cpu().numpy()
                img = imgs[idx]

                f, (ax1, ax2) = plt.subplots(2, 1)

                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                ax1.imshow(img.permute(1, 2, 0).cpu().numpy())

                ax1.axis('off')

                ax2.imshow(out)
                ax2.axis('off')

                f.savefig('{}/{}.png'.format(folder, img_names[0]))
                plt.close(f)


if __name__ == "__main__":



    imgs_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_jd_lidar.TEST_DIR, "imgs/")
    imPts_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_jd_lidar.TEST_DIR, "imPts/")
    lidar_fov_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_jd_lidar.TEST_DIR, "lidar_fov/")
    sparse_depth_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_jd_lidar.TEST_DIR, "sparse_depth/")

    data_loader_val=get_dataloaders(config_jd_lidar.BATCH_SIZE, imgs_root, imPts_root, lidar_fov_root, sparse_depth_root, split=False, val_size=0.20, n_samples=None)


    model = models.get_model_by_name(config_jd_lidar.MODEL)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    folder = '{}_{}_eval_results_depth_completion_{}'.format(constants.INFERENCE_RESULTS,
                                                             config_jd_lidar.MODEL, timestamp)

    predict(model, data_loader_val, config_jd_lidar.MODEL_WEIGHTS_FILENAME, folder)
