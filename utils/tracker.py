import torch
import config
import numpy as np

start = config.NUM_STUFF_CLASSES + config.MAX_DETECTIONS
stop = config.MAX_DETECTIONS*(config.NUM_FRAMES+2)
trk_ids = torch.tensor(list(range(start, stop)), device=config.DEVICE)
ids_bank = torch.tensor(list(range(start, stop)))

# print("trk_ids", trk_ids)
trk_ids_dict = {}


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


def compute_iou_val(prev_box, new_box, prev_label, new_label):

    if new_label != prev_label:
        return 0

    else:
        # min x of intersection
        xA = max(prev_box[0], new_box[0])
        # min y of intersection
        yA = max(prev_box[1], new_box[1])

        # max x of intersection
        xB = min(prev_box[2], new_box[2])
        # max y of intersection
        yB = min(prev_box[3], new_box[3])
        # compute the area of intersection rectangle
        if xA > xB or yA > yB:
            return 0

    return bb_intersection_over_union(new_box, prev_box)


def get_iou_matrix(prev_boxes, new_boxes, prev_labels, new_labels):

    iou_matrix = torch.zeros(
        new_boxes.shape[0], prev_boxes.shape[0]).to(config.DEVICE)
    for i, new_box in enumerate(new_boxes):
        for j, prev_box in enumerate(prev_boxes):
            iou_matrix[i][j] = compute_iou_val(
                prev_box, new_box, prev_labels[j], new_labels[i])
    return iou_matrix


# def init_tracker0(new_det):
#     ids = {"ids": torch.tensor(
#         [torch.tensor(idx + config.NUM_STUFF_CLASSES + config.MAX_DETECTIONS, device=config.DEVICE) for idx, _ in enumerate(new_det["labels"])]).to(config.DEVICE)}
#     # print("ids_0", ids)
#     new_det.update(ids)
#     return new_det


def init_tracker(new_det):
    count = torch.tensor([1], device=config.DEVICE)

    ids_arr = torch.tensor([torch.tensor(idx + start) for idx, _ in enumerate(new_det["labels"])])

    new_ids_obj = {"1": ids_arr.to(config.DEVICE)}
    
    ids = {"ids": ids_arr.to(config.DEVICE)}
    active_ids = {"active": ids_arr}
    count_obj = {"count": count}

    trk_ids_dict.update(new_ids_obj)
    trk_ids_dict.update(active_ids)
    trk_ids_dict.update(count_obj)
    
    new_det.update(ids)
    return new_det


def get_tracked_objects(prev_det, new_det, iou_threshold):



    if len(new_det["boxes"]) > config.MAX_DETECTIONS:

        new_det["masks"] = new_det["masks"][0:config.MAX_DETECTIONS]
        new_det["boxes"] = new_det["boxes"][0:config.MAX_DETECTIONS]
        new_det["labels"] = new_det["labels"][0:config.MAX_DETECTIONS]

        new_det["scores"] = new_det["scores"][0:config.MAX_DETECTIONS]
        if "ids" in new_det.keys():
            new_det["ids"] = new_det["ids"][0:config.MAX_DETECTIONS]
        else:
            new_det["ids"] = torch.zeros_like(new_det["labels"])[
                0:config.MAX_DETECTIONS]



    if prev_det is None:
        return init_tracker(new_det)

    prev_boxes = prev_det["boxes"]
    new_boxes = new_det["boxes"]

    prev_labels = prev_det["labels"]

    new_labels = new_det["labels"]

    if len(prev_boxes) == 0:
        return init_tracker(new_det)

    ids = {"ids": torch.zeros_like(new_det["labels"])}

    # Initialize current frame's ids
    new_det.update(ids)

    # If no new boxes, return 
    if len(new_boxes) == 0:
        return new_det

    # Calculate iou matrix
    iou_matrix = get_iou_matrix(prev_boxes, new_boxes, prev_labels, new_labels)
    

    active_ids = trk_ids_dict["active"]
    # ids not used 
    available_ids = np.setdiff1d(ids_bank, active_ids)

    for idx, _ in enumerate(new_boxes):

        # Get matches for the current detection
        row = iou_matrix[idx, :]
        max_val = torch.max(row)
        max_val_idx = torch.argmax(row)

        # Update this box id 
        if "ids" in prev_det.keys():

            if max_val > iou_threshold:
                new_det["ids"][idx] = prev_det["ids"][max_val_idx]
            else:
                new_det["ids"][idx] = available_ids[0]
                available_ids = available_ids[1:]

    current_frame = trk_ids_dict["count"][0] + 1

    # Update ids used in this frame 
    active_ids_arr = torch.tensor([], device=config.DEVICE)

    if current_frame >= config.NUM_FRAMES + 1:
        for n in range(config.NUM_FRAMES - 1):
            trk_ids_dict["{}".format(n+1)] = trk_ids_dict["{}".format(n+2)]
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

        trk_ids_dict["{}".format(config.NUM_FRAMES)] = new_det["ids"]

        # update active ids
        active_ids_arr = [*active_ids_arr, *new_det["ids"]]
        
    else:
        # Increment count of frames
        trk_ids_dict["count"][0] = current_frame
        new_ids_obj = {"{}".format(current_frame): new_det["ids"]}
        trk_ids_dict.update(new_ids_obj)
        for n in range(current_frame):
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

        # active_ids_arr = torch.tensor([*trk_ids_dict["{}".format(current_frame - 1)], *new_det["ids"]])

    active_ids_arr = torch.unique(torch.tensor(active_ids_arr))
    active_obj = {"active": active_ids_arr}
    trk_ids_dict.update(active_obj)

    return new_det
