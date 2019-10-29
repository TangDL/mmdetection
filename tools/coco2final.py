
import json
import os


json_path = "/data1/gzx/train_round2/val_coco.json"
save_json_path = "/data1/gzx/train_round2/val_coco_final.json"

with open(json_path, 'rt') as f:
    info = json.load(f)

imgs = info["images"]
imgs_name = {}
for img in imgs:
    imgs_name[img["id"]] = img["file_name"]
print(len(imgs_name))

anns = info["annotations"]
print(len(anns))

result = []
for ann in anns:
    res = {}
    id = int(ann["image_id"])
    print(id)
    res["name"] = imgs_name[id]
    res["category"] = int(ann["category_id"])
    x1 = ann["bbox"][0]
    y1 = ann["bbox"][1]
    x2 = ann["bbox"][0]+ann["bbox"][2]
    y2 = ann["bbox"][1]+ann["bbox"][3]
    res["bbox"] = [x1, y1, x2, y2]
    res["score"] = 1
    result.append(res)

json.dump(result, open(save_json_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)