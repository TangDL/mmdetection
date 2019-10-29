import json
import pandas
import numpy as np
import os
from PIL import Image



def get_dense(defces):

    # get the dense iamge id-------------------------------------------------------------------------------------------
    num_img = 2795
    per_num_defces = np.zeros(num_img)
    anns = defces["annotations"]
    for ann in anns:
        img_id = ann["image_id"]
        per_num_defces[img_id]+=1

    dense_imgs = list(np.where(per_num_defces>100))           # index of dense image
    print(dense_imgs)

    # get dense images name------------------------------------------------------------------------/mmdetection/configs/cascade_rcnn_r101_fpn_1x.py
    dense_imgs_name = []
    images = defces["images"]
    for dense_img in dense_imgs[0]:
        image = images[dense_img]
        dense_imgs_name.append(image["file_name"])
    print(dense_imgs_name)

    return dense_imgs, dense_imgs

def remake_sparse_json(defces,dense_imgs):
    # remake and save sparse json file--------------------------------------------------------------------"/mmdetection/configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py"
    anns_sparse = []
    json_sparse_path = "/mmdetection/data1/gzx/train_round2/train_coco_sparse.json"
    anns = defces["annotations"]
    for ann in anns:
        img_id = ann["image_id"]
        if img_id not in dense_imgs[0]:
            anns_sparse.append(ann)
    defces["annotations"] = anns_sparse
    with open(json_sparse_path, 'w') as fp:
        json.dump(defces, fp)

def remake_dense_json(defces,dense_imgs):
    # remake and save dense json file,(comment the front module before use this one)----------------------------------------------------------------------
    anns_dense = []
    json_dense_path = "/mmdetection/data1/gzx/train_round2/train_coco_dense.json"
    anns = defces["annotations"]
    for ann in anns:
        img_id = ann["image_id"]
        if img_id in dense_imgs[0]:
            anns_dense.append(ann)
    defces["annotations"] = anns_dense
    with open(json_dense_path, 'w') as fp:
        json.dump(defces, fp)

def copy_remove_dense_image():
    #copy and remove dense images------------------------------------------------------------------------------------------------------------------------------
    # root_path = "/mmdetection/data1/gzx/train_round2/train_viz/"
    # copy_path = "/mmdetection/data1/gzx/train_round2/dense_image/"
    # for dense_img_name in dense_imgs_name:
    #     img_path = os.path.join(root_path, dense_img_name)
    #     img = Image.open(img_path)
    #     img.save(copy_path+dense_img_name)
    #     os.remove(img_path)
    pass


def analysis_result():
    # aim to compare the defect info between inference and ground truth-----------------------------------------------------------------------------------------
    inference_json_path = '/mmdetection/result.json'
    ground_json_path = "/data1/gzx/train_round2/val_coco_final.json"
    with open(inference_json_path, 'r') as fp:
        inference_json = json.load(fp)
    with open(ground_json_path,'r') as fp:
        ground_json = json.load(fp)

    inference_info = np.zeros(16)
    ground_info = np.zeros(16)
    for defect in inference_json:
        cla = defect["category"]
        inference_info[cla] = inference_info[cla]+1
    for defect in ground_json:
        cla = defect["category"]
        ground_info[cla] += 1
    return inference_info, ground_info

def delete_1():
    ground_json_path = "/data1/gzx/train_round2/train_coco_sparse.json"
    delete_1_json_path = "/data1/gzx/train_round2/train_coco_sparse_delete_1.json"
    with open(ground_json_path,'r') as fp:
        ground_json = json.load(fp)
    annos = ground_json["annotations"]
    annos_new = []
    for anno in annos:
        if anno["category_id"]!=1.0:
            annos_new.append(anno)
    ground_json["annotatons"] = annos_new
    with open(delete_1_json_path, 'w') as fp:
        json.dump(ground_json, fp)

def save_1():
    ground_json_path = "/data1/gzx/train_round2/train_coco_sparse.json"
    save_1_json_path = "/data1/gzx/train_round2/train_coco_sparse_save_1.json"
    with open(ground_json_path, 'r') as fp:
        ground_json = json.load(fp)
    annos = ground_json["annotations"]
    annos_new = []
    for anno in annos:
        if anno["category_id"] == 1.0:
            annos_new.append(anno)
    ground_json["annotatons"] = annos_new
    with open(save_1_json_path, 'w') as fp:
        json.dump(ground_json, fp)


if __name__=='__main__':
    # json_path = "/mmdetection/data1/gzx/train_round2/train_coco.json"
    # with open(json_path, 'r') as fp:
    #     defces = json.load(fp)

    # get the inference and ground info--------------------------------------------------------------------------------------------------------
    # inference_info, ground_info = analysis_result()
    # print(inference_info, '\n', ground_info)

    # delet 1 class-----------------------------------------------------------------------------------------
    delete_1()
    save_1()