# encoding=utf-8
from __future__ import print_function
#from tqdm import tqdm
import numpy as np
import pandas as pd
import json


def main():
    

    #json_path1 = "/mmdetection/result_1.json"
    #json_path2 = "/mmdetection/result_0.json"       # higher Acc than model_1
    #anno_path = "/mmdetection/2_model.json"
    #nms_path = "/mmdetection/nms_2_model.json"
    #final_result_path = "/mmdetection/result.json"
    
    json_path1 = "/data1/DTL/mkimage/mmdetection/result_1.json"
    json_path2 = "/data1/DTL/mkimage/mmdetection/result_0.json"       # higher Acc than model_1
    anno_path = "/data1/DTL/mkimage//mmdetection/2_model.json"
    nms_path = "/data1/DTL/mkimage/mmdetection/nms_2_model.json"
    final_result_path = "/data1/DTL/mkimage/mmdetection/result.json"

    combine_json(json_path1, json_path2, anno_path)
    all_nms_defc = json_average_nms(anno_path)
    json.dump(all_nms_defc, open(nms_path, 'wt'))
    opt_kpAcc(nms_path, json_path2, final_result_path)


def iou(box1, box2):
    area = ((box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)) + ((box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1))
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    iou = (w * h) / (area - (w * h))
    assert iou >= 0
    return iou


def combine_json(json_path1, json_path2, combine_save_path):
    with open(json_path1, 'rt') as f1:
        defc_list1 = json.load(f1)
    with open(json_path2, 'rt') as f2:
        defc_list2 = json.load(f2)
    all_defc_list = defc_list1 + defc_list2
    json.dump(all_defc_list, open(combine_save_path,'wt'))


def json_average_nms(annos_file, IoU_threshold=0.5, score_threshold=0.1):
    defc_list = pd.read_json(open(annos_file), 'r')
    img_reserved = []
    name_list = defc_list["name"].unique()
    for img_name in name_list:
        img_annos = defc_list[defc_list["name"] == img_name]

        img_temp_t = []
        for i in range(len(img_annos)):
            flag = 1
            for j in range(len(img_temp_t)):
                iou_value = iou(img_temp_t[j - 1]["bbox"], img_annos.iloc[i]["bbox"])
                # if iou_value > 0.8:
                if iou_value > IoU_threshold and img_temp_t[j - 1]["category"] == img_annos.iloc[i]["category"]:
                    if img_temp_t[j - 1]["score"] < img_annos.iloc[i]["score"]:
                        img_annos.iloc[i]["score"] = str(img_temp_t[j - 1]["score"] + img_annos.iloc[i]["score"] / 2)
                        rect1 = img_annos.iloc[i]["bbox"]
                        rect2 = img_temp_t[j - 1]["bbox"]
                        x1 = (rect1[0] + rect2[0]) / 2
                        y1 = (rect1[1] + rect2[1]) / 2
                        x2 = (rect1[2] + rect2[2]) / 2
                        y2 = (rect1[3] + rect2[3]) / 2
                        rect = [float(x1), float(y1), float(x2), float(y2)]
                        img_annos.iloc[i]["bbox"] = rect
                        img_temp_t[j - 1] = img_annos.iloc[i]
                    flag = 0
                    break
            if flag == 1:
                if img_annos.iloc[i]["score"] > score_threshold:
                    img_temp_t.append(img_annos.iloc[i])
        for k, item in enumerate(img_temp_t):
            img_temp = {}
            img_temp["name"] = str(item["name"])
            img_temp["bbox"] = (item["bbox"])
            img_temp["category"] = int(item["category"])
            img_temp["score"] =float(item["score"])
            img_reserved.append(img_temp)
    ## print("the numbers of reserved defects is :", len(img_reserved))
    return img_reserved


def opt_kpAcc(json_file1, json_file2, final_save_path):
    '''
        json_file1 is the lower Acc json file need to optimize Acc
        json_file2 is the higher Acc json file used to reference Acc
        final_save is the final result to save file.
    '''
    with open(json_file1, 'rt') as f:
        defc_list1 = json.load(f)
    defc_list2 = pd.read_json(open(json_file2), 'r')
    name_list1 = defc_list2["name"].unique()
    final_defc = []
    for img_name in name_list1:
        for j in range(len(defc_list1)):
            if defc_list1[j]["name"] == img_name:
                final_defc.append(defc_list1[j])
    json.dump(final_defc, open(final_save_path, 'wt'))


if __name__ == '__main__':
    main()


