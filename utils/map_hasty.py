"""This module produces a new annotation json file according the match between the images found
in img_folder and the annotations found in the hasty annotation file"""
import copy
from os import walk
import os.path
import json
import config


def __map_ann_ids(anns, bg_anns=None, obj_anns=None):
    new_anns = []
    new_bg_anns = []
    new_obj_anns = []
    for idx, ann in enumerate(anns):
        ann_id = ann["id"]
        # Set bg ann id
        if bg_anns is not None and len(bg_anns) > 0:
            bg_ann = list(filter(lambda x, annId=ann_id: x["id"] == annId, bg_anns))
            if len(bg_ann) > 0:
                bg_ann = {**ann, "id": idx + 1}
                new_bg_anns.append(bg_ann)
        # Set obj ann id
        if obj_anns is not None and len(obj_anns) > 0:
            obj_ann = list(filter(lambda x, annId=ann_id: x["id"] == annId, obj_anns))
            if len(obj_ann) > 0:
                obj_ann = {**ann, "id": idx + 1}
                new_obj_anns.append(obj_ann)

        ann = {**ann, "id": idx + 1}
        new_anns.append(ann)

    return new_anns, new_bg_anns, new_obj_anns



def get_split(img_folder, output_file):
    """Produdce and save new annotation json file"""
    with open(config.HASTY_COCO_ANN) as hasty_file:
        # read file
        data = json.load(hasty_file)
        # list of images
        images = data["images"]
        annotations = data['annotations']
        categories = data["categories"]
        # get background categories
        bg_categories = list(filter(lambda x: x["supercategory"] == "background", categories))
        bg_categories_ids = list(map(lambda x: x["id"], bg_categories))
        # get object categories
        obj_categories = list(filter(lambda x: x["supercategory"] == "object", categories))
        obj_categories_ids = list(map(lambda x: x["id"], obj_categories))
        # create new json which will contain only the images that are annotated
        new_dict = copy.deepcopy(data)
        # remove all images and annotations
        new_dict['images'] = []
        new_dict['annotations'] = []
        # bg_dict and obj_dict contain only bg and obj annotations respectively
        bg_dict = copy.deepcopy(new_dict)
        obj_dict = copy.deepcopy(new_dict)

        bg_dict['categories'] = bg_categories
        obj_dict['categories'] = obj_categories
        dir_filenames = []

        for(_, _, filenames) in walk(img_folder):
            dir_filenames.extend(filenames)

        if len(dir_filenames) == 0:
            raise AssertionError("{} is empty".format(img_folder))

        for filename in dir_filenames:
            img = list(filter(lambda x, file_name=filename: x['file_name'] == file_name, images))
            img_id = img[0]['id']
            anns = list(filter(lambda x, image_id=img_id: x['image_id'] == image_id, annotations))

            # background annotations
            anns_background = list(filter(lambda x: x['category_id'] in bg_categories_ids, anns))
            if len(anns_background) > 0:
                bg_dict['images'].append(img[0])
                bg_dict['annotations'].extend(anns_background)

            # object annotations
            anns_obj = list(filter(lambda x: x['category_id'] in obj_categories_ids, anns))
            if len(anns_obj) > 0:
                obj_dict['images'].append(img[0])
                obj_dict['annotations'].extend(anns_obj)

            new_dict['images'].append(img[0])
            new_dict['annotations'].extend(anns)

        new_anns, bg_anns, obj_anns = __map_ann_ids(new_dict['annotations'], bg_dict['annotations'], obj_dict['annotations'])
        new_dict['annotations'] = new_anns
        bg_dict['annotations'] = bg_anns
        obj_dict['annotations'] = obj_anns

    filename, file_extension = os.path.splitext(output_file)
    bg_filename = filename + "_bg" + file_extension
    obj_filename = filename + "_obj" + file_extension

    with open(output_file, 'w') as f:
        json.dump(new_dict, f)

    with open(bg_filename, 'w') as f_bg:
        json.dump(bg_dict, f_bg)

    with open(obj_filename, 'w') as f_obj:
        json.dump(obj_dict, f_obj)
