import torch
import config

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_iou_matrix(prev_boxes, new_boxes):

    iou_matrix = torch.zeros(new_boxes.shape[0], prev_boxes.shape[0]).to(config.DEVICE)
    for i, new_box in enumerate(new_boxes):
        for j, prev_box in enumerate(prev_boxes):
            iou_matrix[i][j] = bb_intersection_over_union(new_box, prev_box)
    return iou_matrix


def init_tracker(new_det):
    ids = {"ids": torch.tensor(
        [torch.tensor(idx + config.NUM_STUFF_CLASSES + config.MAX_DETECTIONS).to(config.DEVICE) for idx, _ in enumerate(new_det["labels"])]).to(config.DEVICE)}
    # print(ids)
    new_det.update(ids)
    return new_det


def get_tracked_objects(prev_det, new_det, iou_threshold):



    if len(new_det["boxes"]) > config.MAX_DETECTIONS:

        new_det["masks"] = new_det["masks"][0:config.MAX_DETECTIONS]
        new_det["boxes"] = new_det["boxes"][0:config.MAX_DETECTIONS]
        new_det["labels"] = new_det["labels"][0:config.MAX_DETECTIONS]
        new_det["ids"] = new_det["ids"][0:config.MAX_DETECTIONS]
        new_det["scores"] = new_det["scores"][0:config.MAX_DETECTIONS]



    
    if prev_det is None:
        return init_tracker(new_det)
    
    prev_boxes = prev_det["boxes"]
    new_boxes = new_det["boxes"]

    if len(prev_boxes) == 0:
        return init_tracker(new_det)


    ids = {"ids": torch.zeros_like(new_det["labels"])}
    
    new_det.update(ids)

    if len(new_boxes) == 0:
        return new_det

    iou_matrix = get_iou_matrix(prev_boxes, new_boxes)
    # print(iou_matrix)
    max_id = torch.max(prev_det["ids"])

    for idx, _ in enumerate(new_boxes):

        row = iou_matrix[idx, :]
        # print(row)
        max_val = torch.max(row)
        max_val_idx = torch.argmax(row)

        if "ids" in prev_det.keys():

            if max_val > iou_threshold:

                new_det["ids"][idx] = prev_det["ids"][max_val_idx]
            else:
                new_det["ids"][idx] = max_id + 1
                max_id = max_id + 1

    return new_det
