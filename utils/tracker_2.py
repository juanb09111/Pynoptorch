import torch
import config
import numpy as np
import numba
from numba import jit, cuda 
# import time

start = config.NUM_STUFF_CLASSES + config.MAX_DETECTIONS
stop = config.MAX_DETECTIONS*(config.NUM_FRAMES+2)
ids_bank = torch.tensor(list(range(start, stop)))
trk_ids_dict = {}

threadsperblock = 128


# @cuda.jit
# def bb_intersection_over_union(boxA, boxB):
#     # iou_bb_start = time.time_ns()
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = torch.tensor([max(boxA[0], boxB[0])], device=config.DEVICE)
#     yA = torch.tensor([max(boxA[1], boxB[1])], device=config.DEVICE)
#     xB = torch.tensor([min(boxA[2], boxB[2])], device=config.DEVICE)
#     yB = torch.tensor([min(boxA[3], boxB[3])], device=config.DEVICE)
#     # compute the area of intersection rectangle
#     interArea = torch.tensor([max(0, xB[0] - xA[0] + 1)], device=config.DEVICE)[0] * torch.tensor([max(0, yB[0] - yA[0] + 1)], device=config.DEVICE)[0]
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = torch.tensor([boxA[2] - boxA[0] + 1], device=config.DEVICE)[0] * torch.tensor([boxA[3] - boxA[1] + 1], device=config.DEVICE)[0]
#     boxBArea = torch.tensor([boxB[2] - boxB[0] + 1], device=config.DEVICE)[0] * torch.tensor([boxB[3] - boxB[1] + 1], device=config.DEVICE)[0]
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = torch.tensor([interArea], device=config.DEVICE) / torch.tensor([boxAArea + boxBArea - interArea], device=config.DEVICE)
#     # return the intersection over union value
#     # iou_bb_stop = time.time_ns()
#     # iou_bb_time = iou_bb_stop - iou_bb_start
#     # print("ratio: ", ciou_time/iou_bb_time)
#     # print("iou_bb: ", iou_bb_stop - iou_bb_start)
#     return iou

# @cuda.jit
# def compute_iou_val(prev_box, new_box, prev_label, new_label, iou_matrix, i, j):
#     # ciou_start = time.time_ns()
#     if new_label != prev_label:
#         # ciou_time = 0
#         return 0

#     else:
#         # min x of intersection
#         xA = max(prev_box[0], new_box[0])
#         # min y of intersection
#         yA = max(prev_box[1], new_box[1])

#         # max x of intersection
#         xB = min(prev_box[2], new_box[2])
#         # max y of intersection
#         yB = min(prev_box[3], new_box[3])
#         # compute the area of intersection rectangle
#         if xA > xB or yA > yB:
#             # ciou_time = 0
#             return 0
#     # ciou_stop = time.time_ns()
#     # ciou_time = ciou_stop - ciou_start
#     # print("ciou: ", ciou_stop - ciou_start)
#     # return bb_intersection_over_union(new_box, prev_box)
#     # return 0.6
#     iou_matrix[i][j] = 0.6

@cuda.jit
def get_iou_matrix(iou_matrix, prev_boxes, new_boxes, prev_labels, new_labels):

    for i, new_box in enumerate(new_boxes):
        for j, prev_box in enumerate(prev_boxes):
            if new_labels[i] != prev_labels[j]:
                # ciou_time = 0
                iou_matrix[i][j] = 0

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
                    # ciou_time = 0
                    iou_matrix[i][j] = 0
                else:
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    boxAArea = (new_box[2] - new_box[0] + 1) * (new_box[3] - new_box[1] + 1)
                    boxBArea = (prev_box[2] - prev_box[0] + 1) * (prev_box[3] - prev_box[1] + 1)
                    iou = interArea / float(boxAArea + boxBArea - interArea)
                    iou_matrix[i][j] = iou



def init_tracker(num_ids):

    # Init count of frames
    count = torch.tensor([1], device=config.DEVICE)

    # Init ids for first frame
    ids_arr = torch.tensor([torch.tensor(idx + start) for idx in range(num_ids)], device=config.DEVICE)
    
    # Removre info from other frames if they exist
    for n in range(config.NUM_FRAMES):
        trk_ids_dict.pop("{}".format(n+1), None)

    # Frames count start from 1
    new_ids_obj = {"1": ids_arr}

    # Populate active ids 
    active_ids = {"active": ids_arr.cpu().data.numpy()}

    count_obj = {"count": count}

    trk_ids_dict.update(new_ids_obj)
    trk_ids_dict.update(active_ids)
    trk_ids_dict.update(count_obj)

    return ids_arr


def get_tracked_objects(prev_boxes, new_boxes, prev_labels, new_labels, iou_threshold):

    # start_time = time.time_ns()
    new_ids= torch.zeros(min(config.MAX_DETECTIONS, len(new_boxes)), dtype=torch.long, device=config.DEVICE)

    max_trk = min(config.MAX_DETECTIONS, len(new_boxes))

    if prev_boxes is None:
        return init_tracker(max_trk) 
    
    
    if len(prev_boxes) == 0:
        return init_tracker(max_trk)

    if len(new_boxes) == 0:
        return new_ids

    # iou_start = time.time_ns() 
    # Calculate iou matrix
    iou_matrix = torch.zeros(new_boxes[:max_trk].shape[0], prev_boxes.shape[0], device=config.DEVICE)
    blockspergrid = (len(prev_boxes)*len(new_boxes[:max_trk]) + (threadsperblock - 1)) // threadsperblock
    get_iou_matrix[blockspergrid, threadsperblock](iou_matrix, prev_boxes, new_boxes[:max_trk], prev_labels, new_labels)
    # iou_matrix = get_iou_matrix(prev_boxes, new_boxes[:max_trk], prev_labels, new_labels)
    # iou_stop = time.time_ns()
    # iou_time = iou_stop - iou_start

    # Ids not used yet

    # ids_start = time.time_ns()
    available_ids = np.setdiff1d(ids_bank, trk_ids_dict["active"])

    # current frame
    current_frame = trk_ids_dict["count"][0] + 1

    # prev ids 
    prev_ids = trk_ids_dict["{}".format(current_frame - 1)]

    for idx in range(max_trk):

        # Get matches for the current detection
        row = iou_matrix[idx, :]
        max_val = torch.max(row)
        max_val_idx = torch.argmax(row)

        if max_val > iou_threshold:
            new_ids[idx] = prev_ids[max_val_idx]
        else:
            new_ids[idx] = available_ids[0]
            available_ids = available_ids[1:]
    
    # ids_stop = time.time_ns()
    # ids_time = ids_stop - ids_start

    # update_start = time.time_ns()
    # Update ids used in this frame, this contains all the ids used in the last config.NUM_FRAMES frames
    active_ids_arr = torch.tensor([], device=config.DEVICE)

    # Update frames FIFO

    # if current frame is larger than the maximum number of frames before recycling ids
    if current_frame >= config.NUM_FRAMES + 1:
        for n in range(config.NUM_FRAMES - 1):
            trk_ids_dict["{}".format(n+1)] = trk_ids_dict["{}".format(n+2)]
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

        trk_ids_dict["{}".format(config.NUM_FRAMES)] = new_ids

        # update active ids
        active_ids_arr = [*active_ids_arr, *new_ids]
    
    else:
        # Increment count of frames
        trk_ids_dict["count"][0] = current_frame
        new_ids_obj = {"{}".format(current_frame): new_ids}
        trk_ids_dict.update(new_ids_obj)
        for n in range(current_frame):
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

    # Update trk_ids_dict["active"]
    active_ids_arr = torch.unique(torch.tensor(active_ids_arr))
    active_obj = {"active": active_ids_arr}
    trk_ids_dict.update(active_obj)

    # update_stop = time.time_ns()
    # update_time = update_stop - update_start

    # stop_time = time.time_ns()
    # total_trk = stop_time - start_time
    # print("total tracker", total_trk)
    # print(iou_time*100/total_trk, ids_time*100/total_trk, update_time*100/total_trk)
    return new_ids
