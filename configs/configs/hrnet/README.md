

our team use two model in the preliminary are:
 **cascade_rcnn_hrnetv2p_w32** and **cascade_rcnn_x101_64x4d_fpn** 
 all of them are in the mmdetection tool box, I will describe how to use them following
 ---
### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch 1.1 or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) 4.9 or higher
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

### install  dependency package
after you get this package,just run (other dependencies will be installed automatically)
```
python setup.py develop
```

### train
1.train the **cascade_rcnn_hrnetv2p_w32**  
find the **train.py** in the ..\mmdetection\tools, and config its config file:**..\mmdetection\configs\hrnet\cascade_rcnn_hrnetv2p_w32_20e.py**  
you need to set the iamge filepath and json filepath in dict "data" in the file  
you need to set the "load_from" to the path that model where is,and i put it in ../mmdetection/model  
you need to set the "work_dir" to the path that you want to save model and log file   
you need to download pretrained model from [here](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth)

2.train the **cascade_rcnn_x101_64x4d_fpn**  
a.find the **train.py** in the ..\mmdetection\tools, and config its config file:**..\mmdetection\configs\cascade_rcnn_x101_64x4d_fpn_1x.py**  
b.you need to set the iamge filepath and json filepath in dict "data" in the file  
c.you need to set the "load_from" to the path that model where is,and i put it in ../mmdetection/model  
d.you need to set the "work_dir" to the path that you want to save model and log file  
you need to download pretrained model from [here](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_rcnn_hrnetv2p_w48_20e_20190810-f40ed8e1.pth)

### inference
if you want to inference, find the ..\mmdetection\tools\inference_final.py file, and change some set   
**first**, change the model that you need in the inference, to do this, set the "config_file"and "checkpoint_file"right, i make
the model file in ..\mmdetection\model ,and the "epoch_7"is for cascade_rcnn_x101_64x4d_fpn, "epoch_23" is for cascade_rcnn_hrnetv2p_w32
because the model is too large to update to, i delete them, you need to train by youself
**second**, set the filepath that you want to save the result json, "json_name" and "json_name_temp",the front one is the
final result json path, the other one just for convenience    
**third**, change the inference image path in "test_path"
**finally**, you can change the GPU in "os.environ['CUDA_VISIBLE_DEVICES']" and make sure the "add_filp" is False
all this done, click run   
**note**:two model get two json file, you need to use combine_json.py in tools to combine them, i put mine final json file in the /禾思众成/submit


### Open source information
1.[mmdetection](https://github.com/open-mmlab/mmdetection)  
our model is basis on this, and we get Summary information in [this file](https://github.com/open-mmlab/mmdetection),   
and also learn how to install this powful toolbox in the [installation file](https://github.com/open-mmlab/mmdetection/blob/master/docs/INSTALL.md)   
and how to make model work in [getting_start](https://github.com/open-mmlab/mmdetection/blob/master/docs/GETTING_STARTED.md)   
2.[布匹疵点智能识别方案介绍及baseline(acc:85%左右,mAP:52%左右)](https://tianchi.aliyun.com/notebook-ai/detail?postId=74264)  
It's a good tutorial too, we consider to add anchor ratio in **cascade_rcnn_x101_64x4d_fpn**,set anchor_ratio to [0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 50.0]
,and change the sampler of rcnn to OHEMSampler, and get the origin inference_final.py file
3.[布匹疵点智能识别检测baseline[ 2019广东工业智造创新大赛 赛场一]](https://tianchi.aliyun.com/notebook-ai/detail?postId=71381) we 
alse get some idea from this blog

 
 