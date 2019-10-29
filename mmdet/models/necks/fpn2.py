# encoding=utf-8
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule


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

        for i in range(self.start_level, self.backbone_end_level):   # 0~4， in_channels=[256, 512, 1024, 2048]
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
        if add_extra_convs and extra_levels >= 1:                                 # 不进入
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
    def init_weights(self):                                            # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs, inputs1):                                         # inputs就是输入的4个特征图
        assert len(inputs) == len(self.in_channels)
        assert len(inputs1) == len(self.in_channels)

        # build laterals
        laterals = [                                                   # 建立侧路层，侧路层的每层输出都是256，一共4层
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)        # 4层
        ]
        laterals1 = [                                                  # 模板层，建立侧路层，侧路层的每层输出都是256，一共4层
            lateral_conv(inputs1[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)        # 4层
        ]

        # build top-down path                                      # 就是要想办法在这里融合
        used_backbone_levels = len(laterals)                       # 4层
        for i in range(used_backbone_levels - 1, 0, -1):           # 建立top-down层，这里应该是下采样，和原文的上采样不同
            laterals[i - 1] += F.interpolate(                      # 下采样之后合并，就得到了原文中提到的P层
                laterals[i], scale_factor=2, mode='nearest')       # 这里的操作很奇怪？？？

        # 模板层的top-down
        for i in range(used_backbone_levels - 1, 0, -1):            # 建立top-down层，这里应该是下采样，和原文的上采样不同
            laterals1[i - 1] += F.interpolate(                      # 下采样之后合并，就得到了原文中提到的P层
                laterals1[i], scale_factor=2, mode='nearest')       # 这里的操作很奇怪？？？

        # 融合方式1：直接特征图相加
        for i in range(used_backbone_levels - 1, 0 , -1):
            laterals[i-1] += laterals1[i-1]

        # 融合方式2：特征图的通道数叠加，再用1*1卷积恢复到原来的通道数


        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) # 在P层后面加入3*3的卷积层
        ]
        # part 2: add extra levels                                           # 加入额外的层，应该是文中提及的P6
        if self.num_outs > len(outs):                                        # 条件为真，进入
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:                                     # 额外的卷积层，不进入，
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:                               # 额外的卷积输入层，默认为真，进入
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))  # 多加了一层256的输出层
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):     # range（5,5），不进入
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
