## **环境说明及依赖**

- 所依赖的深度学习框架：PyTorch1.0+

- CUDA版本：10.2
- CUDNN版本：
- 安装依赖包需要运行的命令：
  -  ```sh ./code/mmdetection/compile.sh ```
  -  ` python ./code/mmdetection/setup.py `



## 模型设计

 **模型的整体框架是：Cascade-RCNN+ResNeXt101+FPN+特征融合**

- ResNeXt101：作为backbone，进行图片特征的提取

- FPN+特征融合（***创新点***）：进行瑕疵图片和模板图片的特征融合

- Cascade-RCNN：进行候选框的提取以及候选框的分类和回归

  ### 模型的逻辑设计和步骤说明

  1. 输入瑕疵图片以及瑕疵图片对应的模板图片，对瑕疵图片和模板图片进行resize、padding和归一化
  2. 将进行完预处理之后的瑕疵图片和模板图片输入到ResNeXt101网络中进行特层的提取
  3. ResNeXt101网络大致可分为5个阶段，选取后面4个阶段的最后一层特征层conv2、conv3、conv4、conv5用来进行FPN层的构建
  4. 分别构建出瑕疵图片和模板图片的FPN层，得到 P<sub>defect_i</sub>·和P<sub>template_i</sub> ，其中i={2,3,4,5,6}，
     对于i={2,3,4,5}，采用的融合方式是将对应特征层的通道数进行叠加，比如 P<sub>defect_5</sub>·和P<sub>template_5</sub>的通道数分别为256，将其叠加后得到P<sub>concat_5</sub> 其通道数为512，之后采用1x1的卷积对其进行融合，融合后的特征图P<sub>fused_5</sub> 的通道数为256，尺寸为和输入特征图的尺寸相同。对于P<sub>fused_6</sub> 不采用融合的方式得到，直接令P<sub>defect_i</sub>=P<sub>fused_6</sub> 
  5. 得到特征图后正常使用RPN网络进行候选框提取，并进行分类和回归
  6. 采用Cascade级联的方式不断对候选框进行修正，使其不断逼近真实的瑕疵框坐标

  ### FPN层进行特征融合的详细说明

  ![模型整体框架图](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191027214218692.png)

  

  ​		如图所示，同时输入瑕疵图片和对应的模板图片到网络中，得到 P<sub>defect_i</sub>·和P<sub>template_i</sub> ，其中i={2,3,4,5,6}。取对应的 P<sub>defect_i</sub>·和P<sub>template_i</sub> ，这二者的通道数均为256，且尺寸相同，融合的具体步骤为：

  1.  取对应的P<sub>defect_i</sub>·和P<sub>template_i</sub> ，其中i={2,3,4,5}，将其通道进行叠加，即可得到通道数为512维的特征图P<sub>concat_i</sub> 
  2. 对其进行卷积核大小为1x1，stride为1，padding为0的卷积操作，得到输出通道数为256，尺寸不变的输出特征图P<sub>fused_i</sub>
  3. 对于P<sub>defect_6</sub>和P<sub>template_6</sub> ，直接取P<sub>defect_6</sub>为P<sub>fused_6</sub> ，至此完成整个模板图片特征和瑕疵图片特征的融合操作

  **相应的代码实现**

  ​		FPN层实现模板图片和瑕疵图片融合的代码主要实现在： 

  1. *./code/mmdetection/mmdet/models/cascade_rcnn.py* 

  2. *./code/mmdetection/mmdet/models/fpn.py* 

  
  ## 如何使用
  ### 如何进行训练
  1.训练的主程序入口为./mmdetection/tools/train.py,其中除了可以设定GPU的使用个数和指定GPU训练外，并没有需要进行设定的参数  
  2.进行训练需要指定配置文件，配置文件都集中放在./mmdetection/config文件夹下面  
    **下面将介绍配置文件的详细使用方法：**   
    2.1 在pycharm中开始训练时，直接在运行程序时的“edit configurations"中设置对应的配置文件地址  
    2.2 在配置文件中的model dict中，“type”用来指定模型的整体框架，“pretrained”用来加载初始的预训练模型，若模型不包含在指定路径中，则程序会自动下载。其次  
        其次，可在model dict中设定模型的其他组件。模型的所有组件均放在./mmdetection/mmdet文件夹下。  
    2.3 dataset_type dict进行输入label格式的设定，一般设定为CocoDataset  
    2.4 data dict中进行训练数据集和验证数据集的设置。在train dict中 anno_file和img_file分别指定标签文件和图片文件的路径，值得注意的是，这两个参数可以为列表的形式
        即可以将多个路径同时赋值给该参数  
    2.5 “optimizer”参数进行优化器的指定，其中“lr”在batch为1是设定为0.00125  
    2.6 “total_epochs”进行迭代次数的设定   
    2.7 “work_dir”指定每轮迭代完成后存放模型和日志文件的地址  
    2.8 “load_from”和“resume_from”是开始训练和恢复训练时存放模型的地址，这两者同时有值时，先加载load_from。  
  3.模型中的模块说明：  
    - 模型的所有组成模块都是放在./mmdetection/mmdet文件夹下  
    - 模型可由4部分组成：backbone,neck,head,roi extractor，在GETTING_START文件中有简单的介绍如何自己增加组件的方法  
    - 每个组件的形式都是以class的形式存在的，定义一个新的组件相当于定义一个新的类  
  
 --------
 ### 如何进行预测
 1.预测程序的入口为./mmdetection/tools/inference_final.py  
 2.在预测脚本中，需要进行预测数据集路径的设定，模型的设定，模型配置文件的设定，结果保存路径的设定等  
 3.在预测脚本中，还包含了许多扩展功能，它们都以函数的形式存在，可以方便的进行调用  