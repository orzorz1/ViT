import torch.nn as nn
import torch.nn.functional as F
from modules.DenseUNet_ScaleBlock import Scale

import torch.nn as nn


class ConvBlock(nn.Module):
    '''应用BatchNorm、Relu、bottleneck1x1 Conv2D、3x3 Conv2D，以及可选的dropout
    # 参数
    x: 输入张量
    stage: 密集块的索引
    branch: 每个密集块内的层索引
    nb_filter: 过滤器的数量
    dropout_rate: dropout率
    weight_decay: 权重衰减因子
    '''
    def __init__(self, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        super(ConvBlock, self).__init__()
        self.stage = stage
        self.branch = branch
        self.nb_filter = nb_filter
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.layers_created = False

    def forward(self, x):
        if not self.layers_created:
            self._create_layers(x.shape[1])
            self.layers_created = True

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = self.nb_filter * 4
        x = self.bn1(x)
        x = self.scale1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        if self.dropout_rate:
            x = self.dropout1(x)

        # 3x3 Convolution
        x = self.bn2(x)
        x = self.scale2(x)
        x = self.relu2(x)
        x = self.padding(x)
        x = self.conv2(x)

        if self.dropout_rate:
            x = self.dropout2(x)

        return x

    def _create_layers(self, in_channels):
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.scale1 = Scale()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, self.nb_filter * 4, kernel_size=1, bias=False)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.bn2 = nn.BatchNorm2d(self.nb_filter * 4)
        self.scale2 = Scale()
        self.relu2 = nn.ReLU()
        self.padding = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(self.nb_filter * 4, self.nb_filter, kernel_size=3, bias=False)
        self.dropout2 = nn.Dropout(self.dropout_rate)
