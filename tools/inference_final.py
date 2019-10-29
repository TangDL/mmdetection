# encoding=utf-8
import time, os, sys
import json
import mmcv
from collections import defaultdict
from mmdet.apis import init_detector, inference_detector
import numpy as np
# from tqdm import tqdm
import cv2
import datetime, time
from collections import defaultdict


def get_jpg(path, list1):
    for root, filedir, filenames in os.walk(path):
        for filename in filenames:
            if filename[-4:].upper() == ".JPG" and filename.startswith('template') == 0:
                list1.append(os.path.join(root, filename))


def pick_template(path, template_lsit):
    for root, dirpath, filename in os.walk(path, topdown=True, onerror=None, followlinks=False):
        for name in filename:
            if name.startswith('template') and name.upper().endswith('.JPG'):
                template_lsit.append(os.path.join(root, name))


def fuse_small_big():
    small_json_path = ''
    big_json_path = ''
    total_json_path = ''
    with open(small_json_path, 'r') as fp:
        small_json = json.load(fp)
    with open(big_json_path, 'r') as fp:
        big_json = json.load(fp)
    total_json = small_json + big_json
    with open(total_json_path, 'w') as fp:
        json.dump(total_json, fp, indent=4, separators=(',', ':'))


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


def template_filter(template_result, result):
    Temp1 = defaultdict(list)
    nms_threshold = 0.3
    for template_defect in template_result:
        Temp1[template_defect["name"].split('_')[-1].split('.')[0]].append(
            template_defect["bbox"] + [template_defect["category"]] + [template_defect["score"]])
    for key in Temp1:
        for temp1_defc in Temp1[key]:
            temp1_bbox = temp1_defc[:4]
            for temp2 in result:
                if temp2["name"].split('_')[0] == key and temp2["category"] == temp1_defc[4]:
                    temp2_bbox = temp2["bbox"]
                    # temp1_bbox[2], temp1_bbox[3] = temp1_bbox[0] + temp1_bbox[2], temp1_bbox[1] + temp1_bbox[3]
                    # temp2_bbox[2], temp2_bbox[3] = temp2_bbox[0] + temp2_bbox[2], temp2_bbox[1] + temp2_bbox[3]
                    nms_value = iou(temp1_bbox, temp2_bbox)
                    if nms_value > nms_threshold:
                        result.remove(temp2)
    return result


def parallel_filter(result):
    background_iou_threshold = 0.3
    nms_threshold = 0.7
    k_threshold = 5
    background_defects = []
    # get the background defects first---------------------------------------------------------------------------------
    for i, defect in enumerate(result):
        k_flag = 0
        template_name = defect["name"].split('_')[0]
        temp1_bbox = defect["bbox"]
        # temp1_bbox[2], temp1_bbox[3] = temp1_bbox[0] + temp1_bbox[2], temp1_bbox[1] + temp1_bbox[3]
        for defect_behind in result:
            if defect_behind["name"].split('_')[0] == template_name and \
                    defect_behind["category"] == defect["category"]:
                temp2_bbox = defect_behind["bbox"]
                # temp2_bbox[2], temp2_bbox[3] = temp2_bbox[0] + temp2_bbox[2], temp2_bbox[1] + temp2_bbox[3]
                nms_value = iou(temp1_bbox, temp2_bbox)
                if nms_value > background_iou_threshold:
                    k_flag += 1
        if k_flag > k_threshold:
            background_defects.append(defect)
    print("background_defects number:", len(background_defects))

    # remove background defect in the result---------------------------------------------------------------------------
    for defect0 in result:
        template_name = defect0["name"].split('_')[0]
        temp1_bbox = defect0["bbox"]
        # temp1_bbox[2], temp1_bbox[3] = temp1_bbox[0] + temp1_bbox[2], temp1_bbox[1] + temp1_bbox[3]
        for background_defect in background_defects:
            if background_defect["name"].split('_')[0] == template_name and \
                    background_defect["category"] == defect0["category"]:
                temp2_bbox = background_defect["bbox"]
                # temp2_bbox[2], temp2_bbox[3] = temp2_bbox[0] + temp2_bbox[2], temp2_bbox[1] + temp2_bbox[3]
                nms_value = iou(temp1_bbox, temp2_bbox)
                if nms_value > nms_threshold:
                    result.remove(defect0)
                    break
    return result


def list_nms(defc_list, iou_threhold, min_confidence):
    annos_all = defaultdict(list)
    defect_resverd = []
    for defc in defc_list:
        annos_all[defc["name"]].append(defc["bbox"] + [defc["category"]] + [defc["score"]])
    for key, annos in annos_all.items():  # for every image, anno contain all defects
        img_temp_t = []
        for anno in annos:  # for every defect in one image
            flag = 1
            for j in range(len(img_temp_t)):
                if img_temp_t[j]["category"] == anno[4]:
                    iou_value = iou(img_temp_t[j]["bbox"], anno[0:4])
                    if iou_value > iou_threhold and img_temp_t[j]["score"] < anno[5]:
                        img_temp_t[j] = {"name": key, "bbox": anno[0:4], "category": anno[4], "score": anno[5]}
                        flag = 0
                        break
            if flag == 1 and anno[5] > min_confidence:
                img_temp_t.append({"name": key, "bbox": anno[0:4], "category": anno[4], "score": anno[5]})
        defect_resverd = defect_resverd + img_temp_t
    return defect_resverd


def main():
    # set parameters-------------------------------------------------------------------------------------------

    config_file = "/mmdetection/configs/fpn2.py"
    config_file_1 = "/mmdetection/configs/cascade_rcnn_x101_32x4d_fpn_1x.py"
    # config_file_1 = "/data1/lgj/bupi/round2/cascade_rcnn_x101_32x4d_fpn_1x.py"
    # config_file = "/data1/DTL/mkimage/mmdetection/configs/fpn2.py"

    checkpoint_file = "/mmdetection/models/fpn_epoch_12.pth"
    checkpoint_file_1 = "/mmdetection/models/epoch_12.pth"
    # checkpoint_file_1 = "/data1/lgj/bupi/round2/work_dirs/resnext101_data_aug/epoch_12.pth"
    # checkpoint_file = '/data1/DTL/mkimage/mmdetection/models/fpn_epoch_12.pth'

    test_path = "/tcdata/guangdong1_round2_testB_20191024/"
    template_path = "/tcdata/guangdong1_round2_testB_20191024/"
    # test_path ="/data1/teimage/0917B1_189c867e9c5fa3b72201909170225215/"
    # template_path ="/data1/teimage/0917B1_189c867e9c5fa3b72201909170225215/"

    json_name = '/mmdetection/result.json'
    json_name = '/mmdetection/result_0.json'
    # json_name_1 = '/mmdetection/result_1.json'
    # json_name = '/data1/DTL/mkimage/mmdetection/result_0.json'
    # json_name_1 = '/data1/DTL/mkimage/mmdetection/result_1.json'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model_1 = init_detector(config_file_1, checkpoint_file_1, device='cuda:0')

    # get image list------------------------------------------------------------------------------------------------
    img_list = []
    template_list = []
    get_jpg(test_path, img_list)
    pick_template(template_path, template_list)

    # print("the number of defect image is:", len(img_list))

    result_0, result_1 = [], []
    template_result_0, template_result_1 = [], []

    now = time.time()
    # input template image and make inference------------------------------------------------------------------------
    for i, template_name in enumerate(template_list, 1):
        full_img = template_name
        predict = inference_detector(model, full_img)
        predict_1 = inference_detector(model_1, full_img)
        image_name = template_name.split('/')[-1]
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    template_result_0.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})
        print("the number of template defects is:", len(template_result_0))
        for i, bboxes in enumerate(predict_1, 1):
            if len(bboxes) > 0:
                 defect_label = i
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    template_result_1.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    # defect image inference---------------------------------------------------------------------------------------
    # input img_list and make inference----------------------------------------------------------------------------
    for i, img_name in enumerate(img_list, 1):
        full_img = img_name
        predict = inference_detector(model, full_img)
        # predict_1 = inference_detector(model_1, full_img)
        image_name = img_name.split('/')[-1]
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    result_0.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

        # for i, bboxes in enumerate(predict_1, 1):
        #     if len(bboxes) > 0:
        #         defect_label = i
        #         for bbox in bboxes:
        #             x1, y1, x2, y2, score = bbox.tolist()
        #             x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
        #             result_1.append(
        #                 {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})
    end_time = time.time()
    print("end_time %s S" % (end_time - now))

    # start template filter----------------------------------------------------------------------------------
    result_0 = template_filter(template_result_0, result_0)
    # result_1 = template_filter(template_result_1, result_1)

    # print("the defects number after template filter:", len(result))
    # start parallel filter------------------------------------------------------------------------------------
    # result = parallel_filter(result)
    # print("the defects number after parallel filter:", len(result))

    result_0 = list_nms(result_0, iou_threhold=0.8, min_confidence=0.001)
    # result_1 = list_nms(result_1, iou_threhold=0.8, min_confidence=0.001)

    # compare defects in template image and defect image
    with open(json_name, 'w') as fp:
        json.dump(result_0, fp, indent=4, separators=(',', ': '))
    # with open(json_name_1, 'w') as fp:
    #     json.dump(result_1, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()





