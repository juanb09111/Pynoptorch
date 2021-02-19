"""This module produces a new annotation json file according the match between the images found
in img_folder and the annotations found in the hasty annotation file"""
import copy
from os import walk
import os.path
import json
from datetime import datetime
import math
import uuid 
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


def __add_augmented_img(img, img_anns, aug_filenames, max_img_id, max_ann_id):

    new_img = None
    new_anns = []

    # get img name
    img_fname = ".".join(img['file_name'].split(".")[0:-1])
    
    # get img annotation
    # find match in agumented dataset
    match = list(filter(lambda aug_fname: img_fname in aug_fname, aug_filenames))
    
    # if there's a match
    if len(match) > 0:
        # create new img dict
        new_img ={**img, "id": max_img_id + 1, "file_name": match[0]}

        for ann_dict in img_anns:
            # create new img ann dict
            new_img_ann = {**ann_dict, "id": max_ann_id + 1, "image_id": max_img_id + 1}
            new_anns.append(new_img_ann)
    
    return new_img , new_anns



def get_split(img_folder, output_file, aug_data_set_folder=None):
    """Produce and save new annotation json file"""
    with open(config.HASTY_COCO_ANN) as hasty_file:
        # read file
        data = json.load(hasty_file)
        # list of images
        images = data["images"]
        print("total images in coco_ann: ", len(images))

        annotations = data['annotations']
        max_ann_id = max(list(map(lambda ann: ann["id"], annotations)))

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

        # intialize array with image file names found in the img_folder
        dir_filenames = []

        # intialize arrays
        aug_filenames = [] #agumented images filenames
        images_file_names = [] # images filenames from annotations
        images_in_folder = []  # images in annotations also found in the root folder
        max_n_augmented = 0 # max number of augmented images to be used

        max_img_id = max(list(map(lambda img: img["id"], images)))

        for(_, _, filenames) in walk(img_folder):
            dir_filenames.extend(filenames)

        if aug_data_set_folder != None:
            for (_,_, fnames) in walk(aug_data_set_folder):
                aug_filenames.extend(fnames)

            images_file_names = list(map(lambda x: x['file_name'], images))
            images_in_folder = list(filter(lambda im_filename: im_filename in dir_filenames, images_file_names))
            max_n_augmented = math.floor(len(images_in_folder)*config.AUGMENTED_PERCENT)
        
        if len(dir_filenames) == 0:
            raise AssertionError("{} is empty".format(img_folder))
        
        
        augmented_count = 0
        for filename in dir_filenames:
            # Find an image in the annotation file that matches the current image from the image folder
            img = list(filter(lambda x, file_name=filename: x['file_name'] == file_name, images))
            
            # if found
            if len(img) > 0:

                # get img id
                img_id = img[0]['id']
                # find the annotations for the current image
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


                
                # Do likewise for augmented dataset, do this inside this if so it is guaranteed that there's an annotation for this image
                if len(aug_filenames) > 0 and augmented_count <= max_n_augmented:

                    aug_img, aug_anns = __add_augmented_img(img[0], anns, aug_filenames, max_img_id, max_ann_id)
                    
                    aug_anns_background = list(filter(lambda x: x['category_id'] in bg_categories_ids, aug_anns))
                    if len(anns_background) > 0:
                        bg_dict['images'].append(aug_img)
                        bg_dict['annotations'].extend(aug_anns_background)
                    
                    aug_anns_obj = list(filter(lambda x: x['category_id'] in obj_categories_ids, aug_anns))
                    if len(anns_obj) > 0:
                        obj_dict['images'].append(aug_img)
                        obj_dict['annotations'].extend(aug_anns_obj)
                    
                    new_dict['images'].append(aug_img)
                    new_dict['annotations'].extend(aug_anns)

                    augmented_count = augmented_count + 1
                    max_img_id = max_img_id + 1
                    max_ann_id = max_ann_id + 1
                    

        print("aug n: ", augmented_count)
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
