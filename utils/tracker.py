import torch
import config
import temp_variables
import numpy as np
import numba
from numba import jit, cuda
import operator
from utils import lucas_kanade
import time

start = config.NUM_STUFF_CLASSES + config.MAX_DETECTIONS
stop = config.MAX_DETECTIONS*(config.NUM_FRAMES+2)
ids_bank = torch.tensor(list(range(start, stop)))
trk_ids_dict = {}

threadsperblock = 128



@cuda.jit
def get_iou_matrix(iou_matrix, prev_boxes, new_boxes, prev_labels, new_labels):

    for i, new_box in enumerate(new_boxes):
        for j, prev_box in enumerate(prev_boxes):
                
            # ciou_time = 0
            iou_matrix[i][j] = 0

            if new_labels[i] == prev_labels[j]:
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
    count = torch.tensor([1], device=temp_variables.DEVICE)

    # Init ids for first frame
    ids_arr = torch.tensor([torch.tensor(idx + start) for idx in range(num_ids)], device=temp_variables.DEVICE)
    
    # Remove info from other frames if they exist
    for n in range(config.NUM_FRAMES):
        trk_ids_dict.pop("{}".format(n+1), None)

    # Frames count start from 1
    new_ids_obj = {"1": ids_arr}

    # Populate active ids 
    active_ids = {"active": ids_arr.cpu().data.numpy()}

    count_obj = {"count": count}

    # Init unmatched
    unmatched_obj = {"unmatched_{}".format(1): {}}

    trk_ids_dict.update(new_ids_obj)
    trk_ids_dict.update(active_ids)
    trk_ids_dict.update(count_obj)
    trk_ids_dict.update(unmatched_obj)

    return ids_arr


def map_to_super_class(label, super_cat_indices):

    if label in super_cat_indices:
        return config.NUM_STUFF_CLASSES + config.NUM_THING_CLASSES + 2

    return label

def flatten_unmatched(trk_ids_dict):

    dict_keys = list(trk_ids_dict.keys())
    dict_keys = [ key for key in dict_keys if "unmatched_" in key]

    all_boxes=torch.tensor([], device=temp_variables.DEVICE)
    all_labels=torch.tensor([], device=temp_variables.DEVICE, dtype=torch.long)
    all_ids=torch.tensor([], device=temp_variables.DEVICE, dtype=torch.long)
    all_masks=torch.tensor([], device=temp_variables.DEVICE)

    for key in dict_keys:

        obj = trk_ids_dict[key]

        if len(list(obj.keys())) > 0:
            boxes = trk_ids_dict[key]["boxes"]
            labels = trk_ids_dict[key]["labels"]
            ids = trk_ids_dict[key]["ids"]
            masks = trk_ids_dict[key]["masks"]

            all_boxes = torch.cat((all_boxes, boxes))
            all_labels = torch.cat((all_labels, labels))
            all_ids = torch.cat((all_ids, ids))
            all_masks = torch.cat((all_masks, masks))
    
    return all_boxes, all_masks, all_labels, all_ids


def update_unmatched(prev_img_fname, next_image_fname):
    timestamp= time.time()

    dict_keys = list(trk_ids_dict.keys())
    dict_keys = [ key for key in dict_keys if "unmatched_" in key]

    count = trk_ids_dict["count"]

    for key in dict_keys:
        
        frame_num = int(key.split("_")[-1])
        obj = trk_ids_dict[key]

        # apply lucas kanade to all prev frames except last one (bacause it's already updated)      
        if len(list(obj.keys())) > 0:
            
            boxes = trk_ids_dict[key]["boxes"]
            print("type 2", type(boxes))
            masks = trk_ids_dict[key]["masks"]

            pred_boxes, pred_masks = lucas_kanade.lucas_kanade_per_mask(
                    prev_img_fname,
                    next_image_fname,
                    masks,
                    boxes,
                    config.INFERENCE_CONFIDENCE,
                    find_keypoints=True,
                    save_as="{}_unmatched_{}".format(timestamp, frame_num))

            #update frame
            trk_ids_dict[key]["boxes"] = pred_boxes
            trk_ids_dict[key]["masks"] = pred_masks



def get_tracked_objects(prev_boxes, prev_masks, new_boxes, prev_labels, new_labels, super_cat_indices, iou_threshold, prev_img_fname, next_image_fname):

    # Update prev bboxes
    update_unmatched(prev_img_fname, next_image_fname)
    

    new_ids= torch.zeros(min(config.MAX_DETECTIONS, len(new_boxes)), dtype=torch.long, device=temp_variables.DEVICE)

    max_trk = min(config.MAX_DETECTIONS, len(new_boxes))

    if prev_boxes is None:
        return init_tracker(max_trk) 
    
    
    if len(prev_boxes) == 0:
        return init_tracker(max_trk)

    if len(new_boxes) == 0:
        return new_ids

    # Flatteng unmatched obj
    # print("trk_ids_dict", trk_ids_dict)
    unmatched_boxes, unmatched_masks, unmatched_labels, unmatched_ids = flatten_unmatched(trk_ids_dict)
    # print(unmatched_boxes, unmatched_labels, unmatched_ids)

    prev_boxes_extended = torch.cat((prev_boxes, unmatched_boxes))
    prev_labels_extended = torch.cat((prev_labels, unmatched_labels))

    #TODO: Find prev ids from the latest frame
    
    prev_ids = trk_ids_dict["{}".format(trk_ids_dict["count"].item())][:len(prev_boxes)]
    print("prev_ids", prev_ids)
    print("unmatched: ", unmatched_ids)
    prev_ids_extended = torch.cat((prev_ids, unmatched_ids))
    # Calculate iou matrix

    iou_matrix = torch.zeros(new_boxes[:max_trk].shape[0], prev_boxes_extended.shape[0], device=temp_variables.DEVICE)
    blockspergrid = (len(prev_boxes_extended)*len(new_boxes[:max_trk]) + (threadsperblock - 1)) // threadsperblock

    mapped_prev_labels_extended = torch.tensor(list(map(lambda val: map_to_super_class(val, super_cat_indices), prev_labels_extended)),  device=temp_variables.DEVICE)
    mapped_new_labels = torch.tensor(list(map(lambda val: map_to_super_class(val, super_cat_indices), new_labels)),  device=temp_variables.DEVICE)

    get_iou_matrix[blockspergrid, threadsperblock](iou_matrix, prev_boxes_extended, new_boxes[:max_trk], mapped_prev_labels_extended, mapped_new_labels)

 
    # Ids not used yet

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
            # there are ids coming from prev frame and then ids from prev unmatched boxes there need to be an if
            
            if max_val_idx < len(prev_boxes):
                new_ids[idx] = prev_ids[max_val_idx]
            else:
                # it's an unmatched box from a prev frame
                print("matched an unmatched!!", prev_ids_extended[max_val_idx])
                #TODO: If matched with an unmatched box, that unmatched box needs to be popped out
                new_ids[idx] = prev_ids_extended[max_val_idx]
        else:
            new_ids[idx] = available_ids[0]
            available_ids = available_ids[1:]

    
    # concat prev_ids and new_ids
    combined = torch.cat((prev_ids, new_ids))
    # print(combined)
    # Find uniques
    uniques, counts = combined.unique(return_counts=True)
    # print("size", uniques, counts)
    # Find values that appear only once
    difference = uniques[counts == 1]
    # print("difference: ", difference)
    
    # indices of values in prev_ids that are not in new_ids
    unmatched_indices = [i for i, x in enumerate(prev_ids[:len(prev_boxes)]) if x in difference]
    unmatched_ids = prev_ids[unmatched_indices]
    unmatched_boxes = prev_boxes[unmatched_indices]
    unmatched_labels = prev_labels[unmatched_indices]
    unmatched_masks = prev_masks[unmatched_indices]
    unmatched_obj = {"ids": unmatched_ids, "boxes": unmatched_boxes, "labels": unmatched_labels, "masks": unmatched_masks}

    print("len: ", len(prev_ids[:len(prev_boxes)]), len(prev_boxes), len(prev_boxes[unmatched_indices]))
    

    # Update ids used in this frame, this contains all the ids used in the last config.NUM_FRAMES frames
    active_ids_arr = torch.tensor([], device=temp_variables.DEVICE)

    # Update frames FIFO


    ## -------------------------------
    if current_frame > config.NUM_FRAMES + 1:
        for n in range(config.NUM_FRAMES - 1):
            # update unmatched 
            trk_ids_dict["unmatched_{}".format(n+1)] = trk_ids_dict["unmatched_{}".format(n+2)]

      
        trk_ids_dict["unmatched_{}".format(config.NUM_FRAMES)] = unmatched_obj

    
    else:

        new_unmatched_obj = {"unmatched_{}".format(current_frame - 1): unmatched_obj}

        trk_ids_dict.update(new_unmatched_obj)


    ## ----------------------------------------------

    # if current frame is larger than the maximum number of frames before recycling ids
    if current_frame >= config.NUM_FRAMES + 1:
        for n in range(config.NUM_FRAMES - 1):
            trk_ids_dict["{}".format(n+1)] = trk_ids_dict["{}".format(n+2)]
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]

            # update unmatched 
            # trk_ids_dict["unmatched_{}".format(n+1)] = trk_ids_dict["unmatched_{}".format(n+2)]

        trk_ids_dict["{}".format(config.NUM_FRAMES)] = new_ids
        # trk_ids_dict["unmatched_{}".format(config.NUM_FRAMES)] = unmatched_obj

        # update active ids
        active_ids_arr = [*active_ids_arr, *new_ids]
    
    else:
        # Increment count of frames
        trk_ids_dict["count"][0] = current_frame

        new_ids_obj = {"{}".format(current_frame): new_ids}
        # new_unmatched_obj = {"unmatched_{}".format(current_frame - 1): unmatched_obj}

        trk_ids_dict.update(new_ids_obj)
        # trk_ids_dict.update(new_unmatched_obj)
        for n in range(current_frame):
            active_ids_arr = [*active_ids_arr, *trk_ids_dict["{}".format(n+1)]]
    # Update trk_ids_dict["active"]
    active_ids_arr = torch.unique(torch.tensor(active_ids_arr))
    active_obj = {"active": active_ids_arr}
    trk_ids_dict.update(active_obj)


    return new_ids, unmatched_indices, unmatched_ids
