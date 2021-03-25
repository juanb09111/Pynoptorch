import config
import temp_variables
import constants
import models

from utils.tensorize_batch import tensorize_batch

import os
import os.path
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


from utils import get_datasets, lucas_kanade
from utils.show_segmentation import apply_semantic_mask_gpu, apply_instance_masks, save_fig, draw_bboxes, apply_panoptic_mask_gpu
from utils.panoptic_fusion import threshold_instances, sort_by_confidence, threshold_overlap, panoptic_canvas, panoptic_fusion, get_stuff_thing_classes
from utils.tracker import get_tracked_objects, init_tracker
import time 

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

temp_variables.DEVICE = device
torch.cuda.empty_cache()

all_categories, stuff_categories, thing_categories = get_stuff_thing_classes()

things_names = list(map(lambda thing: thing["name"], thing_categories))
super_cat_indices = [things_names.index(
    label) + 1 for label in config.SUPER_CLASS]

color_max_val = 1

if config.BOUNDING_BOX_ONLY:
    color_max_val = 255


def visualize_results(im, summary_batch, next_det, fname):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale = 0.4
    fontColor = (color_max_val, color_max_val, color_max_val)
    lineType = 1
    text = ""
    if summary_batch:
        summary = summary_batch[0]
        for obj in summary:
            text = text + \
                '{}: {}'.format(obj["name"], obj["count_obj"]) + " - "

    im = draw_bboxes(im,
                     next_det["boxes"],
                     thing_categories,
                     color_max_val=color_max_val,
                     labels=next_det["labels"],
                     ids=next_det["ids"])

    save_fig(im, "lucas_kanade/res_5", "{}".format(fname),
             tensor_image=False, cv_umat=True)

def visualize_results_untracked(im, summary_batch, next_det, untracked_det, fname):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (30, 30)
    fontScale = 0.4
    fontColor = (color_max_val, color_max_val, color_max_val)
    lineType = 1
    text = ""
    if summary_batch:
        summary = summary_batch[0]
        for obj in summary:
            text = text + \
                '{}: {}'.format(obj["name"], obj["count_obj"]) + " - "

    im = draw_bboxes(im,
                     next_det["boxes"],
                     thing_categories,
                     color_max_val=color_max_val,
                     labels=next_det["labels"],
                     ids=next_det["ids"])
    
    im = draw_bboxes(im,
                     untracked_det["boxes"],
                     thing_categories,
                     color_max_val=color_max_val,
                     labels=untracked_det["labels"],
                     ids=untracked_det["ids"],
                     box_color = (255,255,0))

    save_fig(im, "lucas_kanade/res_6", "{}".format(fname),
             tensor_image=False, cv_umat=True)


def view_masks(model,
               data_loader_val,
               num_classes,
               weights_file,
               result_type,
               folder,
               confidence=0.5):

    # Create folde if it doesn't exist
    Path(folder).mkdir(parents=True, exist_ok=True)
    # load weights
    model.load_state_dict(torch.load(weights_file))
    # move model to the right device
    model.to(device)
    frame = 0
    for image, next_image, anns, next_anns in data_loader_val:
        # print(anns, next_anns, images[0].shape, next_images[0].shape)
        # print(image[0].shape)
        if next_image[0] is not None:
            images = list([image[0], next_image[0]])
            images = tensorize_batch(images, device)
            file_names = list(
                [anns[0]["file_name"], next_anns[0]["file_name"]])
            print(".....{}......".format(file_names[1]))
            model.eval()
            with torch.no_grad():
                # model_start = time.time()
                outputs = model(images)
                # model_end = time.time()
                # print("model_time: ", model_end-model_start)

                # Threshold preds----------
                # thr_start = time.time()
                
                threshold_preds = threshold_instances(
                    outputs, threshold=config.CONFIDENCE_THRESHOLD)
                threshold_preds = threshold_overlap(
                    threshold_preds, nms_threshold=config.NMS_THRESHOLD)
                sorted_preds = sort_by_confidence(threshold_preds)
                # thr_end = time.time()

                # print("thr time: ", thr_end-thr_start)
                # Lucas kanade-----------


                prev_img_fname = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), config.TEST_DIR, file_names[0])
                next_image_fname = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), config.TEST_DIR, file_names[1])
                prev_masks = sorted_preds[0]["masks"]
                prev_boxes = sorted_preds[0]["boxes"]

                # lk_start = time.time()
                pred_boxes, pred_masks = lucas_kanade.lucas_kanade_per_mask(
                    prev_img_fname,
                    next_image_fname,
                    prev_masks,
                    prev_boxes,
                    config.INFERENCE_CONFIDENCE,
                    find_keypoints=True,
                    save_as="{}_{}".format(frame, frame+1))
                # lk_end = time.time()
                # print("lk time: ", lk_end - lk_start)
                # update prev boxes with lk-predicted boxes
                sorted_preds[0]["boxes"] = pred_boxes
                sorted_preds[0]["masks"] = pred_masks

                # Object Tracking----------
                prev_det = sorted_preds[0]
                next_det = sorted_preds[1]

                # obj_trck_start = time.time() 
                tracked_obj, untracked_indices, untracked_ids = get_tracked_objects(
                    prev_det["boxes"],
                    prev_det["masks"],
                    next_det["boxes"],
                    prev_det["labels"],
                    next_det["labels"],
                    super_cat_indices,
                    config.OBJECT_TRACKING_IOU_THRESHHOLD,
                    prev_img_fname,
                    next_image_fname)
                # obj_trck_end = time.time()
                # print("obj_track: ", obj_trck_end - obj_trck_start)
                next_det["ids"] = tracked_obj

                # untracked objects
                untracked_obj = {}
                untracked_boxes = {"boxes": prev_det["boxes"][untracked_indices]}
                untracked_labels = {"labels": prev_det["labels"][untracked_indices]}
                untracked_ids = {"ids": untracked_ids}

                untracked_obj.update(untracked_boxes)
                untracked_obj.update(untracked_labels)
                untracked_obj.update(untracked_ids)

                if len(tracked_obj) > 0:
                    next_det["boxes"] = next_det["boxes"][:len(
                        tracked_obj)]
                    next_det["masks"] = next_det["masks"][:len(
                        tracked_obj)]
                    next_det["scores"] = next_det["scores"][:len(
                        tracked_obj)]
                    next_det["labels"] = next_det["labels"][:len(
                        tracked_obj)]

                if result_type == "panoptic" and len(next_det["masks"]) > 0:

                    # Get intermediate prediction and semantice prediction
                    inter_pred_batch, sem_pred_batch, summary_batch = panoptic_fusion(
                        [next_det], all_categories, stuff_categories, thing_categories, threshold_by_confidence=False, sort_confidence=False)
                    canvas = panoptic_canvas(
                        inter_pred_batch, sem_pred_batch, all_categories, stuff_categories, thing_categories)[0]

                    if canvas is None:
                        im = images[1].cpu().permute(1, 2, 0).numpy()
                    else:

                        im = apply_panoptic_mask_gpu(
                            images[1], canvas).cpu().permute(1, 2, 0).numpy()

                    # Visualize results --------
                    # visualize_results(im, summary_batch, next_det, "{}".format(frame+1))
                    visualize_results_untracked(im, summary_batch, next_det, untracked_obj, "{}".format(frame+1))
            # if config.BOUNDING_BOX_ONLY:
            #     im = next_image[0].cpu().permute(1, 2, 0).numpy()
            #     im = draw_bboxes(im,
            #                      next_det["boxes"],
            #                      thing_categories,
            #                      color_max_val=color_max_val,
            #                      labels=next_det["labels"],
            #                      ids=next_det["ids"])
            #     save_fig(im, "lucas_kanade/res", "{}".format(frame+1),     =False, cv_umat=True)

            frame = frame +1
            # torch.cuda.empty_cache()
                # return next_image[0], None, next_det
                # plt.imshow(im)

            #     if result_type == "panoptic":
            #         panoptic_fusion.get_panoptic_results(
            #             images, outputs, all_categories, stuff_categories, thing_categories, folder, file_names)
            #         torch.cuda.empty_cache()
            #     else:
            #         for idx, output in enumerate(outputs):
            #             file_name = file_names[idx]
            #             if result_type == "instance":
            #                 im = apply_instance_masks(images[idx], output['masks'], 0.5)

            #             elif result_type == "semantic":
            #                 logits = output["semantic_logits"]
            #                 mask = torch.argmax(logits, dim=0)
            #                 im = apply_semantic_mask_gpu(images[idx], mask, config.NUM_STUFF_CLASSES + config.NUM_THING_CLASSES)

            #             save_fig(im, folder, file_name)

            #             torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    test_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), config.TEST_DIR)
    data_loader_test = get_datasets.get_dataloaders(
        config.BATCH_SIZE, test_dir, is_test_set=True, return_next=True)

    print(len(data_loader_test))
    confidence = 0.5
    model = models.get_model()

    if config.INFERENCE_OBJECT_TRACKING:
        init_tracker(config.MAX_DETECTIONS)

    if config.INSTANCE:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "instance",
                   '{}{}_{}_results_instance_{}'.format(constants.INFERENCE_RESULTS,
                                                        config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.SEMANTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "semantic",
                   '{}{}_{}_results_semantic_{}'.format(constants.INFERENCE_RESULTS,
                                                        config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)

    if config.PANOPTIC:
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        view_masks(model, data_loader_test, config.NUM_THING_CLASSES + config.NUM_THING_CLASSES,
                   config.MODEL_WEIGHTS_FILENAME,
                   "panoptic",
                   '{}{}_{}_results_panoptic_{}'.format(constants.INFERENCE_RESULTS,
                                                        config.MODEL, config.BACKBONE, timestamp),
                   confidence=0.5)
