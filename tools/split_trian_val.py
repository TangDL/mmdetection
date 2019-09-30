import os
import json
import numpy as np
import glob
import shutil
from labelme import utils
from sklearn.model_selection import train_test_split
np.random.seed(41)

#0为背景
classname_to_id ={
                   # u'破洞': 1, u'水渍': 2, u'油渍': 2, u'污渍': 2, u'三丝': 3, u'结头': 4, u'花板跳': 5, u'百脚': 6, u'毛粒': 7,
                   # u'粗经': 8, u'松经': 9, u'断经': 10, u'吊经': 11, u'粗维': 12, u'纬缩': 13, u'浆斑': 14, u'整经结': 15, u'星跳': 16,
                   # u'跳花': 16, u'断氨纶': 17, u'稀密档': 18, u'浪纹档': 18, u'色差档': 18, u'磨痕': 19, u'轧痕': 19, u'修痕': 19,
                   # u'烧毛痕': 19, u'死皱': 20, u'云织': 20, u'双纬': 20, u'双经': 20, u'跳纱': 20, u'筘路': 20, u'纬纱不良': 20,
                   "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10,
                   "11":11, "12":12, "13":13, "14":14, "15":15, "16":16, "17":17, "18":18, "19":19, "20":20
                   }
class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(label)
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_path = "/bupi_data/bupi_data/train_defect_total/"
    saved_coco_path = "/mmdetection/bupi/"
    # 创建文件
    if not os.path.exists("%sbupi/annotations/"%saved_coco_path):
        os.makedirs("%sbupi/annotations/"%saved_coco_path)
    if not os.path.exists("%sbupi/images/train2017/"%saved_coco_path):
        os.makedirs("%sbupi/images/train2017"%saved_coco_path)
    if not os.path.exists("%sbupi/images/val2017/"%saved_coco_path):
        os.makedirs("%sbupi/images/val2017"%saved_coco_path)
    # 获取images目录下所有的json文件列表
    json_list_path = glob.glob(labelme_path + "/*.json")
    print("json_list_path num: ", len(json_list_path))
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    train_path, val_path = train_test_split(json_list_path, test_size=0.12)
    print("train_n:", len(train_path), 'val_n:', len(val_path))

    # 把训练集转化为COCO的json格式
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%sbupi/annotations/instances_train2017.json'%saved_coco_path)
    # for file in train_path:
    #     shutil.copy(file.replace("json","jpg"),"%sbupi/images/train2017/"%saved_coco_path)
    # for file in val_path:
    #     shutil.copy(file.replace("json","jpg"),"%sbupi/images/val2017/"%saved_coco_path)

    # 把验证集转化为COCO的json格式
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%sbupi/annotations/instances_val2017.json'%saved_coco_path)