# -*- coding: utf-8 -*
import json
from tqdm import tqdm
import numpy as np
import cv2

# ----------------------------------------------------- PARAMETERS -----------------------------------------------------
image_path = "/data1/gzx/train_round2/val_viz/"                                                # images painted gt bbox(pink)
save_path = "/data1/gzx/train_round2/val_viz/"
json_path = "/mmdetection/result.json"                                                            # dete json (yellow)
yellow = 255                                                                                 # 255 to yellow, 0 to red
class_name = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'10', u'11', u'12', u'13', u'14', u'15',
              u'16']
# ----------------------------------------------------------------------------------------------------------------------

with open(json_path, 'rt') as f:
    info = json.load(f)
output_result = info
defc_num = len(output_result)
print("defc_num:", defc_num)
for i in tqdm(range(defc_num)):
    if output_result[i]['name'] == output_result[i-1]['name']:
        img = cv2.imread(save_path + output_result[i-1]['name'])
        rects = output_result[i]['bbox']
        score = str(round(output_result[i]['score'], 2))
        obj = class_name[int(output_result[i]['category'])]
        text = obj + ":" + score
        text.encode("utf-8").decode("utf-8")
        xmin = int(rects[0])
        ymin = int(rects[1])
        xmax = int(rects[2])
        ymax = int(rects[3])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (yellow, 0, 255), 3)
        cv2.putText(img, text, (xmax, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (yellow, 0, 255), 2)
        cv2.imwrite(save_path + output_result[i]['name'], img)
    else:
        img = cv2.imread(image_path + output_result[i]['name'])
        print(np.shape(img))
        obj = class_name[int(output_result[i]['category'])]
        rects = output_result[i]['bbox']
        score = str(round(output_result[i]['score'], 2))
        # print(obj)
        text = obj + ":" + score
        text.encode("utf-8").decode("utf-8")
        xmin = int(rects[0])
        ymin = int(rects[1])
        xmax = int(rects[2])
        ymax = int(rects[3])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (yellow, 0, 255), 3)
        cv2.putText(img, text, (xmax, ymax), cv2.FONT_HERSHEY_COMPLEX, 1, (yellow, 0, 255), 2)
        cv2.imwrite(save_path + output_result[i]['name'], img)