# encoding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import numpy as np
import torch

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,             # [256, 512, 1024, 2048]
                 out_channels,            # 256
                 num_outs,                # 5
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)                          # 4
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins               # 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()                 # 侧路连接层，设定为ModuleList类型，可以往里面添加模块
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):   # 0~4�?in_channels=[256, 512, 1024, 2048]
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,                                               # 卷积核的大小
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)        # 4层[[256,256],[512,256],[1024,256],[2048,256]]
            self.fpn_convs.append(fpn_conv)          # 4层[[256,256],[256,256],[256,256],[256,256]]

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level # 5-4+0=1
        if add_extra_convs and extra_levels >= 1:                                 # 不进�?
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]  # 2048
                else:
                    in_channels = out_channels                                   # 256
                extra_fpn_conv = ConvModule(
                    in_channels,                    # 2048
                    out_channels,                   # 256
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):                                            # 初始化权�?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):                                         # inputs就是输入�?个特征图
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [                                                   # 建立侧路层，侧路层的每层输出都是256，一�?�?
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)        # 4�?
        ]

        # build top-down path                                      # 就是要想办法在这里融�?
        used_backbone_levels = len(laterals)                       # 4�?
        for i in range(used_backbone_levels - 1, 0, -1):           # 建立top-down层，这里应该是下采样，和原文的上采样不同
            laterals[i - 1] += F.interpolate(                      # 下采样之后合并，就得到了原文中提到的P�?
                laterals[i], scale_factor=2, mode='nearest')       # 这里的操作很奇怪？？？

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) # 在P层后面加�?*3的卷积层
        ]
        # part 2: add extra levels                                           # 加入额外的层，应该是文中提及的P6
        if self.num_outs > len(outs):                                        # 条件为真，进�?
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:                                     # 额外的卷积层，不进入�?
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:                               # 额外的卷积输入层，默认为真，进入
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))  # 多加了一�?56的输出层
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):     # range�?,5），不进�?
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



@NECKS.register_module
class FPN2(nn.Module):

    def __init__(self,
                 in_channels,             # [256, 512, 1024, 2048]
                 out_channels,            # 256
                 num_outs,                # 5
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)                          # 4
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins               # 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()                 # 侧路连接层，设定为ModuleList类型，可以往里面添加模块
        self.fpn_convs = nn.ModuleList()

        self.fuse_conv = ConvModule(                                                # 定义卷积�?
            512,
            256,
            1,
            # stride=1,
            # padding=1,
            conv_cfg=None,
            norm_cfg=None,
            # norm_cfg=dict(type='BN', requires_grad=True),                # add normalization
            activation=self.activation,
            inplace=False)

        for i in range(self.start_level, self.backbone_end_level):   # 0~4�?in_channels=[256, 512, 1024, 2048]
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,                                               # 卷积核的大小
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)        # 4层[[256,256],[512,256],[1024,256],[2048,256]]
            self.fpn_convs.append(fpn_conv)          # 4层[[256,256],[256,256],[256,256],[256,256]]

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level # 5-4+0=1
        if add_extra_convs and extra_levels >= 1:                                 # 不进�?
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]  # 2048
                else:
                    in_channels = out_channels                                   # 256
                extra_fpn_conv = ConvModule(
                    in_channels,                    # 2048
                    out_channels,                   # 256
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):                                            # 初始化权�?
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs, inputs1):                                         # inputs就是输入�?个特征图
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [                                                   # 建立侧路层，侧路层的每层输出都是256，一�?�?
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)        # 4�?
        ]
        laterals1 = [                                                  # 模板层，建立侧路层，侧路层的每层输出都是256，一�?�?
            lateral_conv(inputs1[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)        # 4�?
        ]

        # build top-down path                                      # 就是要想办法在这里融�?
        used_backbone_levels = len(laterals)                       # 4�?
        for i in range(used_backbone_levels - 1, 0, -1):           # 建立top-down层，这里进行了上采样
            laterals[i - 1] += F.interpolate(                      # 上采样之后合并，就得到了原文中提到的P�?
                laterals[i], scale_factor=2, mode='nearest')       # 这里的操作很奇怪？？？

        # 改进方法2：直接在top-down时进行融�?------------------------改进2, got bad effect
        # for i in range(used_backbone_levels-1, 0, -1):
        #     laterals[i-1]=laterals[i-1]+F.interpolate(laterals[i]+laterals1[i], scale_factor=2, mode='nearest')
        # # -------------------------------------------------------

        # 模板层的top-down
        for i in range(used_backbone_levels - 1, 0, -1):            # 建立top-down�?
            laterals1[i - 1] += F.interpolate(                      # 下采样之后合并，就得到了原文中提到的P�?
                laterals1[i], scale_factor=2, mode='nearest')       # 这里的操作很奇怪？？？从上到下，尺寸增�?

        # # 改进3：只进行�?层的融合（采用直接相加的方式�?-------------改进3
        # laterals[0] = laterals[0] + laterals1[0]
        # # --------------------------------------------------------------

        # #融合方式1：直接特征图相加--------------------------------融合1
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     laterals[i-1] += laterals1[i-1]
        # #----------------------------------------------------------
        # 融合方式2：特征图的通道数叠加，再用1*1卷积恢复到原来的通道�?->融合的最终目的是增加网络对模板的识别�?---------融合2
        # 问题�?.还没有实现cascade预训练模型的加载-->新增加了一层卷积层，和原本的模型参数对应不�?
        # 改进的方向：1.现在只是�?个特征层上实现融合，可以尝试�?个特征层上尝试融�?
        #            2.现在的融合方式是两张图片各自进行top-down，能否top-down的过程中融合�?
        #              即，laterals[i-1] = laterals[i] + F.interpolate(laterals[i]+laterals1[i]) 或者各自进行插值后再相�?
        #            3.可以尝试只是进行某一层的统合，比如laterals[0],或者是laterals[4](最后一�?
        #            4.change 1*1 conv to 3*3 conv
        #            5.only make some levels into fuse
        for i in range(0, used_backbone_levels):                               # 将模板图片和瑕疵图片的通道合并
            laterals[i] = torch.cat((laterals[i], laterals1[i]), 1)         # only make 0-2 level into fuse
        laterals2 = [                                                       # 将合并后的特征层进行卷积
            self.fuse_conv(laterals[i])
            for i in range(0, used_backbone_levels)                            # 4�?
            ]
        laterals = laterals2
        # ------------------------------------------------------------------------------------------------------------

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) # 在P层后面加�?*3的卷积层
        ]
        # part 2: add extra levels                                           # 加入额外的层，应该是文中提及的P6
        if self.num_outs > len(outs):                                        # 条件为真，进�?
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:                                     # 额外的卷积层，不进入�?
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:                               # 额外的卷积输入层，默认为真，进入
                    orig = inputs[self.backbone_end_level - 1]               # 瑕疵图片在fpn中的最后一层特征层
                    # 改进方法1：在5个特征层上均采用融合-----------------------------------------------------------改进1
                    # 改进方法3：若该段单独使用，则是改进方�?-----------------------------------------------------改进3
                    # orig1 = inputs1[self.backbone_end_level - 1]           # 模板图片在fpn中的最后一层特征层
                    # orig = orig+orig1                                      # 采用直接相加的方式融合这最后一�?
                    # ------------------------------------------------------------------------------------------------
                    # improvement method 4, make conv in 5 feature map and fuse
                    # orig1 = inputs1[self.backbone_end_level - 1]
                    # orig = torch.cat((orig, orig1), 1)
                    # orig = self.fuse_conv(orig)
                    # # ------------------------------------------------------------------------------------------------
                    outs.append(self.fpn_convs[used_backbone_levels](orig))  # 多加了一�?56的输出层
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))  # 不用输入的额外卷积，则使用侧路层进行额外卷积
                for i in range(used_backbone_levels + 1, self.num_outs):     # range�?,5），不进入。其作用是将层数扩展到num_outs
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
