import torch
import numpy as np
import os.path
import json
import config
import matplotlib.pyplot as plt
from .show_bbox import apply_panoptic_mask


def threshold_instances(preds, threshold=0.5):

    for i in range(len(preds)):
        mask_logits, bbox_pred, class_pred, confidence = preds[i][
            "masks"], preds[i]["boxes"], preds[i]["labels"], preds[i]["scores"]

        indices = (confidence > threshold).nonzero().view(1, -1)

        mask_logits = torch.index_select(mask_logits, 0, indices.squeeze(0))
        bbox_pred = torch.index_select(bbox_pred, 0, indices.squeeze(0))
        class_pred = torch.index_select(class_pred, 0, indices.squeeze(0))
        confidence = torch.index_select(confidence, 0, indices.squeeze(0))

        preds[i]["masks"] = mask_logits
        preds[i]["boxes"] = bbox_pred
        preds[i]["labels"] = class_pred
        preds[i]["scores"] = confidence

    return preds


def sort_by_confidence(preds):

    sorted_preds = [{"masks": torch.zeros_like(preds[i]["masks"]),
                     "boxes": torch.zeros_like(preds[i]["boxes"]),
                     "labels": torch.zeros_like(preds[i]["labels"]),
                     "scores": torch.zeros_like(preds[i]["scores"]),
                     "semantic_logits": torch.zeros_like(preds[i]["semantic_logits"])} for i, _ in enumerate(preds)]

    for i in range(len(preds)):

        sorted_indices = torch.argsort(preds[i]["scores"])

        for idx, k in enumerate(sorted_indices):
            sorted_preds[i]["masks"][idx] = preds[i]["masks"][k]
            sorted_preds[i]["boxes"][idx] = preds[i]["boxes"][k]
            sorted_preds[i]["labels"][idx] = preds[i]["labels"][k]
            sorted_preds[i]["scores"][idx] = preds[i]["scores"][k]

        sorted_preds[i]["semantic_logits"] = preds[i]["semantic_logits"]

    return sorted_preds


def get_stuff_thing_classes():

    ann_file = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", config.HASTY_COCO_ANN)

    with open(ann_file) as hasty_file:
        # read file
        data = json.load(hasty_file)
        all_categories = data["categories"]

    stuff_categories = list(
        filter(lambda cat: cat["supercategory"] != "object", all_categories))

    thing_categories = list(
        filter(lambda cat: cat["supercategory"] == "object", all_categories))

    return all_categories, stuff_categories, thing_categories


def get_MLB(sorted_preds):

    all_categories, _, thing_categories = get_stuff_thing_classes()

    all_ids = list(map(lambda cat: cat["id"], all_categories))

    batch_mlb = []

    # iterate over batch
    for _, pred in enumerate(sorted_preds):

        boxes = pred["boxes"]
        classes = pred["labels"]
        sem_logits = pred["semantic_logits"]

        non_background_objs = len(torch.nonzero(classes))

        mlb_arr = torch.zeros(non_background_objs, *sem_logits[0].shape)

        i = 0
        for j, box in enumerate(boxes):

            # get current box's class
            bbox_class = classes[j]

            # Exclude background class
            if bbox_class > 0:

                # Thing categories do not include background, hence bbox_class - 1
                bbox_cat = thing_categories[bbox_class - 1]

                # Find matching id in all_ids
                cat_idx = all_ids.index(bbox_cat["id"])

                # Get the corresponding semantic tensor for the same class
                matched_sem_logits = sem_logits[cat_idx + 1]

                # Create zeroed-tensor the size of the semantic mask
                mlb = torch.zeros_like(matched_sem_logits)

                xmin, ymin, xmax, ymax = box

                xmin = xmin.to(dtype=torch.long)
                ymin = ymin.to(dtype=torch.long)
                xmax = xmax.to(dtype=torch.long)
                ymax = ymax.to(dtype=torch.long)

                mlb[ymin:ymax, xmin:xmax] = matched_sem_logits[ymin:ymax, xmin:xmax]

                mlb_arr[i] = mlb

                i += 1

        batch_mlb.append(mlb_arr)
    return batch_mlb


def get_MLA(sorted_preds):

    batch_mla = []

    for _, pred in enumerate(sorted_preds):
        classes = pred["labels"]
        masks = pred["masks"]
        boxes = pred["boxes"]

        if masks.shape[0] > 0:

            non_background_objs = len(torch.nonzero(classes))

            mla_arr = torch.zeros(non_background_objs, *
                                torch.squeeze(*masks[0]).shape)

            i = 0

            for j, mask in enumerate(masks):

                if classes[j] > 0:

                    mask_logits = torch.zeros_like(mask[0])

                    box = boxes[j]
                    xmin, ymin, xmax, ymax = box

                    xmin = xmin.to(dtype=torch.long)
                    ymin = ymin.to(dtype=torch.long)
                    xmax = xmax.to(dtype=torch.long)
                    ymax = ymax.to(dtype=torch.long)

                    box_mask_probs = mask[0][ymin:ymax, xmin:xmax]

                    box_mask_logits = torch.log(
                        box_mask_probs) - torch.log1p(-box_mask_probs)

                    mask_logits[ymin:ymax, xmin:xmax] = box_mask_logits

                    mla_arr[i] = mask_logits

                    i += 1
        else:
            mla_arr = []

        batch_mla.append(mla_arr)

    return batch_mla


def fuse_logits(MLA, MLB):

    batch_size = len(MLA)
    fl_batch = []
    for i in range(batch_size):
        
        mla = MLA[i]

        mlb = MLB[i]

        if len(mla) > 0 and len(mlb) > 0:

            sigmoid_mla = torch.sigmoid(mla)
            sigmoid_mlb = torch.sigmoid(mlb)

            fl = (sigmoid_mla + sigmoid_mlb) * (mla + mlb)

            fl_batch.append(fl)
        else:
            fl_batch.append(None)

    return fl_batch


def panoptic_fusion(preds):

    inter_pred_batch = []
    sem_pred_batch = []

    batch_size = len(preds)

    all_categories, _, _ = get_stuff_thing_classes()

    # Get list of cat in the form of (idx, supercategory)
    cat_idx = list(map(lambda cat_tuple: (
        cat_tuple[0], cat_tuple[1]["supercategory"]), enumerate(all_categories)))

    # Filter previous list with supercategory == "background"
    stuff_cat_idx_sup = list(
        filter(lambda cat_tuple: cat_tuple[1] == "background", cat_idx))

    # Get indices only for stuff categories
    stuff_cat_idx = list(
        map(lambda sutff_cat: sutff_cat[0], stuff_cat_idx_sup))

    # Add background class 0
    stuff_cat_idx = [0, *[x + 1 for x in stuff_cat_idx]]

    stuff_cat_idx = torch.LongTensor(stuff_cat_idx).to(config.DEVICE)

    confidence_thresholded = threshold_instances(preds)

    sorted_preds = sort_by_confidence(confidence_thresholded)

    MLA = get_MLA(sorted_preds)

    MLB = get_MLB(sorted_preds)

    fused_logits_batch = fuse_logits(MLA, MLB)

    for i in range(batch_size):
        
        fused_logits = fused_logits_batch[i]
        
        if fused_logits is not None: 

            fused_logits = fused_logits.to(config.DEVICE)
            sem_logits = preds[i]["semantic_logits"]

            # TODO: check if background class 0 needs to be included in stuff_cat_idx
            stuff_sem_logits = torch.index_select(
                sem_logits, 0, stuff_cat_idx)
            # print("suff-fused", stuff_sem_logits.shape, fused_logits.shape)
            # intermediate logits
            inter_logits = torch.cat((stuff_sem_logits, fused_logits))

            # Intermediate Prediction
            inter_pred = torch.argmax(inter_logits, dim=0)

            # plt.figure(i)
            # plt.imshow(inter_pred.cpu().numpy())

            # Semantic Prediction includinf background class 0
            sem_pred = torch.argmax(sem_logits, dim=0)

            # plt.figure(i+10)
            # plt.imshow(sem_pred.cpu().numpy())

            inter_pred_batch.append(inter_pred)

            sem_pred_batch.append(sem_pred)
        else:
            inter_pred_batch.append(None)

            sem_pred_batch.append(None)

    return inter_pred_batch, sem_pred_batch


def map_include(x, classes_arr):

    if x in classes_arr:
        return x
    else:
        return 0


def map_exclude(x, classes_arr):

    if x in classes_arr:
        return 0
    else:
        return x


def panoptic_canvas(inter_pred_batch, sem_pred_batch):

    batch_size = len(inter_pred_batch)

    panoptic_canvas_batch = []

    all_categories, _, _ = get_stuff_thing_classes()

    # print("all_categories", all_categories)

    # Get list of cat in the form of (idx, supercategory)
    cat_idx = list(map(lambda cat_tuple: (
        cat_tuple[0], cat_tuple[1]["supercategory"]), enumerate(all_categories)))

    # Filter previous list with supercategory == "background"
    stuff_cat_idx_sup = list(
        filter(lambda cat_tuple: cat_tuple[1] == "background", cat_idx))

    # Get indices only for stuff categories
    stuff_cat_idx = list(
        map(lambda sutff_cat: sutff_cat[0], stuff_cat_idx_sup))

    # Add background class 0
    stuff_cat_idx = [0, *[x + 1 for x in stuff_cat_idx]]
    # print("stuff_cat_idx", stuff_cat_idx)

    # Stuff classes index in intermediate prediction
    stuff_in_inter_pred_idx = [x for x in range(len(stuff_cat_idx))]
    # print("stuff_in_inter_pred_idx", stuff_in_inter_pred_idx)

    for i in range(batch_size):

        inter_pred = inter_pred_batch[i]
        sem_pred = sem_pred_batch[i]

        if inter_pred == None or sem_pred == None:
            panoptic_canvas_batch.append(None)
        
        else:

            # convert to numpy. Here the width and height are in (0,1) respectively
            inter_pred_np = inter_pred.cpu().numpy()
            sem_pred_np = sem_pred.cpu().numpy()

            # flatten numpy
            inter_pred_flat_np = inter_pred_np.flatten()
            sem_pred_flat_np = sem_pred_np.flatten()

            original_shape = inter_pred.shape
            width = original_shape[1]
            height = original_shape[0]

            # Stuff canvas
            stuff_canvas_list = list(
                map(lambda x: map_include(x, stuff_cat_idx), sem_pred_flat_np))
            stuff_canvas_arr = np.array(stuff_canvas_list)

            # Things canvas
            things_canvas_list = list(map(lambda x: map_exclude(
                x, stuff_in_inter_pred_idx), inter_pred_flat_np))
            things_canvas_arr = np.array(things_canvas_list)

            # panoptic_canvas
            panoptic_canvas_arr = np.where(
                things_canvas_arr == 0, stuff_canvas_arr, things_canvas_arr)
            panoptic_canvas = panoptic_canvas_arr.reshape((height, width))

            # plt.figure(i)
            # plt.imshow(panoptic_canvas)

            panoptic_canvas_batch.append(panoptic_canvas)

    return panoptic_canvas_batch


def get_panoptic_results(images, preds, folder, filenames):

    batch_size = len(preds)

    inter_pred_batch, sem_pred_batch = panoptic_fusion(preds)

    panoptic_canvas_batch = panoptic_canvas(inter_pred_batch, sem_pred_batch)

    height, width = panoptic_canvas_batch[0].shape

    my_path = os.path.dirname(__file__)

    for i in range(batch_size):

        file_name_basename = os.path.basename(filenames[i])
        file_name = os.path.splitext(file_name_basename)[0]

        dppi = 96
        fig, ax = plt.subplots(1, 1, figsize=(
            width/dppi, height/dppi), dpi=dppi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        canvas = panoptic_canvas_batch[i]

        img = images[i].cpu().permute(1, 2, 0).numpy()
        im = apply_panoptic_mask(img, canvas)

        ax.imshow(im,  interpolation='nearest', aspect='auto')
        plt.axis('off')
        fig.savefig(os.path.join(
            my_path, '../{}/{}.png'.format(folder, file_name)))
        plt.close(fig)
