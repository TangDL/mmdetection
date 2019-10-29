
import numpy as np

def main():
    #gen coco pretrained weight
    import torch
    num_classes = 21
    model_coco = torch.load("/mmdetection/TDL/model/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth")

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
                                                         :num_classes]

    # change the anchors number
    # weights
    w = model_coco["state_dict"]["rpn_head.rpn_cls.weight"]
    model_coco["state_dict"]["rpn_head.rpn_cls.weight"] = torch.nn.Parameter(torch.cat([w,w,w]))
    w = model_coco["state_dict"]["rpn_head.rpn_reg.weight"]
    model_coco["state_dict"]["rpn_head.rpn_reg.weight"] = torch.nn.Parameter(torch.cat([w, w, w]))

    # bias
    b = model_coco["state_dict"]["rpn_head.rpn_cls.bias"]
    model_coco["state_dict"]["rpn_head.rpn_cls.bias"] = torch.nn.Parameter(torch.cat([b,b,b]))
    b = model_coco["state_dict"]["rpn_head.rpn_reg.bias"]
    model_coco["state_dict"]["rpn_head.rpn_reg.bias"] = torch.nn.Parameter(torch.cat([b,b,b]))


    # save new model
    torch.save(model_coco, "/mmdetection/TDL/model/cascade_rcnn_r101_coco_pretrained_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()