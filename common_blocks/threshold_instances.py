import torch
import numpy as np
import os.path
import json
import config


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
                     "scores": torch.zeros_like(preds[i]["scores"])} for i, _ in enumerate(preds)]

    for i in range(len(preds)):

        sorted_indices = np.argsort(preds[i]["scores"])

        for idx, k in enumerate(sorted_indices):
            sorted_preds[i]["masks"][idx] = preds[i]["masks"][k]
            sorted_preds[i]["boxes"][idx] = preds[i]["boxes"][k]
            sorted_preds[i]["labels"][idx] = preds[i]["labels"][k]
            sorted_preds[i]["scores"][idx] = preds[i]["scores"][k]

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


def get_MLB(preds):

    all_categories, _, thing_categories = get_stuff_thing_classes()

    all_ids = list(map(lambda cat: cat["id"], all_categories))

    batch_mlb = []

    # iterate over batch
    for _, pred in enumerate(preds):

        boxes = pred["boxes"]
        classes = pred["labels"]
        sem_logits = pred["semantic_logits"]

        non_background_objs = len(torch.nonzero(classes))

        # mlb_arr = torch.zeros(boxes.shape[0], sem_logits.shape[1:])
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

                mlb[ymin: ymax, xmin:xmax] = matched_sem_logits[ymin: ymax, xmin:xmax]

                mlb_arr[i] = mlb

                i += 1

        batch_mlb.append(mlb_arr)
    return batch_mlb


def get_MLA(preds):

    batch_mla = []

    for _, pred in enumerate(preds):

        classes = pred["labels"]
        masks = pred["masks"]

        non_background_objs = len(torch.nonzero(classes))

        mla_arr = torch.zeros(non_background_objs, *masks[0].shape)

        i = 0

        for j, mask in enumerate(masks):

            if classes[j] > 0:

                mla_arr[i] = mask

                i += 1

        batch_mla.append(mla_arr)

    return batch_mla


def fuse_logits(MLA, MLB):

    batch_size = len(MLA)
    fl_batch = []
    for i in range(batch_size):

        # these are probs and it needs to be converted to logits
        mla_probs = MLA[i]
        # convert to logits
        mla_logits = torch.log(mla_probs) - torch.log1p(-mla_probs)

        # mlb comes in logits already
        mlb = MLB[i]

        sigmoid_mla = mla_probs
        sigmoid_mlb = torch.sigmoid(mlb)

        # print(mla_logits.shape, mlb.shape)

        fl = (sigmoid_mla + sigmoid_mlb) * (mla_logits + mlb)

        fl_batch.append(fl)

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

    confidence_thresholded = threshold_instances(preds)

    sorted_preds = sort_by_confidence(confidence_thresholded)

    MLA = get_MLA(sorted_preds)

    MLB = get_MLB(preds)

    fused_logits_batch = fuse_logits(MLA, MLB)

    for i in range(batch_size):

        fused_logits = fused_logits_batch[i]

        sem_logits = preds[i]["semantic_logits"]

        # TODO: check if background class 0 needs to be included in stuff_cat_idx
        stuff_sem_logits = torch.index_select(
            sem_logits, 0, torch.LongTensor(stuff_cat_idx))

        # intermediate logits
        inter_logits = torch.cat((stuff_sem_logits, fused_logits))
        
        # Intermediate Prediction
        inter_pred = torch.argmax(inter_logits, dim=0)

        # Semantic Prediction includinf background class 0
        sem_pred = torch.argmax(sem_logits, dim=0)

        inter_pred_batch.append(inter_pred) 

        sem_pred_batch.append(sem_pred) 

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

    all_categories, _, _ = get_stuff_thing_classes()

    print(all_categories)

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
    print(stuff_cat_idx)

    # Stuff classes index in intermediate prediction
    stuff_in_inter_pred_idx = [x for x in range(len(stuff_cat_idx))]
    print(stuff_in_inter_pred_idx)

    for i in range(batch_size):

        inter_pred = inter_pred_batch[i]
        sem_pred = sem_pred_batch[i]

        original_shape = inter_pred.shape

        inter_pred_flat_np = torch.flatten(inter_pred).numpy()
        sem_pred_flat_np = torch.flatten(sem_pred).numpy()

        # Stuff canvas
        stuff_canvas_arr = list(map(lambda x: map_include(x, stuff_cat_idx), sem_pred_flat_np))

        stuff_canvas = np.asarray(stuff_canvas_arr).reshape(original_shape)

        # Things canvas
        things_canvas_arr = list(map(lambda x: map_exclude(x, stuff_in_inter_pred_idx), inter_pred_flat_np))

        things_canvas = np.asarray(things_canvas_arr).reshape(original_shape)

        print(stuff_canvas.shape)

        




masks = torch.rand(10, 1200, 1920)
confidence = torch.tensor([0, 0, 0, 0, 0, 0.7, 0.9, 0.8, 0, 0])
labels = torch.randint(0, 7, (1, 10)).squeeze(0)
boxes = torch.randint(0, 1200, (10, 4))
sem_logits = torch.rand(8, 1200, 1920)

print(labels)
pred1 = {"masks": masks, "boxes": boxes,
         "labels": labels,
         "scores": confidence,
         "semantic_logits": sem_logits}
pred2 = pred1

preds = [pred1, pred2]
# print(preds[0]["scores"].shape)

# res = threshold_instances(preds)

# print(preds[0]["scores"].shape)

# sorted_preds = sort_by_confidence(preds)


# batch_mlb = get_MLB(preds)

inter_pred_batch, sem_pred_batch = panoptic_fusion(preds)

panoptic_canvas(inter_pred_batch, sem_pred_batch)
