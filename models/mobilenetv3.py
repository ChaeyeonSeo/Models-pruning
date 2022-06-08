# -*- coding: UTF-8 -*-

'''
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''

    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6


class H_swish(nn.Module):
    '''
    hard swish
    '''

    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''

    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y


class Block(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''

    def __init__(self, in_planes, exp_size, out_planes, kernel_size, stride, use_SE, NL):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Block, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'

        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_planes == out_planes)
        self.exp_size = exp_size
        self.in_planes = in_planes

        if exp_size == in_planes:
            # Without expansion, the first depthwise convolution is omitted
            # depthwise convolution
            self.conv1_w = nn.Conv2d(in_planes, exp_size, kernel_size=kernel_size, stride=stride,
                                   padding=(kernel_size - 1) // 2, groups=in_planes, bias=False)
            self.bn1_w = nn.BatchNorm2d(exp_size)
            self.nl_w = nn.ReLU(inplace=True)  # non-linearity
            # self.se_w = nn.Sequential()  # SE module
            # if use_SE:
            #     self.se_w = SEModule(exp_size)
            self.se_w = SEModule(exp_size) if use_SE else nn.Sequential()

            if use_HS:
                self.nl_w = H_swish()
            # Linear Pointwise Convolution
            self.conv2_w = nn.Conv2d(exp_size, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2_w = nn.BatchNorm2d(out_planes)
            self.shortcut_w = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut_w = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )

        else:
            # With expansion
            # Pointwise Convolution for expansion
            self.conv1 = nn.Conv2d(in_planes, exp_size, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=exp_size)
            self.nl1 = nn.ReLU(inplace=True)  # non-linearity
            if use_HS:
                self.nl1 = H_swish()
            # Depthwise Convolution
            self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride,
                                   padding=(kernel_size - 1) // 2, groups=exp_size, bias=False)
            self.bn2 = nn.BatchNorm2d(exp_size)
            self.nl2 = nn.ReLU(inplace=True)  # non-linearity
            if use_HS:
                self.nl2 = H_swish()
            self.se = nn.Sequential()  # SE module
            if use_SE:
                self.se = SEModule(exp_size)
            # Linear Pointwise Convolution
            self.conv3 = nn.Conv2d(exp_size, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            # nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            self.bn3 = nn.BatchNorm2d(out_planes)
            self.shortcut = nn.Sequential()
            if stride == 1 and in_planes != out_planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_planes),
                )

    def forward(self, x, expand=False):
        if self.exp_size == self.in_planes:
            out = self.conv1_w(x)
            out = self.bn1_w(out)
            out = self.nl_w(out)
            # out = self.nl_w(self.bn1_w(self.conv1_w(x)))
            out = self.se_w(out)
            out = self.bn2_w(self.conv2_w(out))

        else:
            out = self.nl1(self.bn1(self.conv1(x)))
            out = self.nl2(self.bn2(self.conv2(out)))
            out = self.se(out)
            out = self.bn3(self.conv3(out))


        if self.use_residual:
            out = out + x
        # if expand:
        #     return out, out1
        # else:
        return out


class MobileNetV3(nn.Module):
    def __init__(self, mode='small', num_classes=10):
        '''
        cfg: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(MobileNetV3, self).__init__()

        mode = mode.lower()
        assert mode in ['large', 'small']

        # setting of the model
        if mode == 'large':
            # Configuration of a MobileNetV3-Large Model
            self.cfg = [
                # kernel_size, expansion, out_planes, SE, NL, stride
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        elif mode == 'small':
            # Configuration of a MobileNetV3-Small Model
            self.cfg = [
                # kernel_size, expansion, out_planes, use_SE, NL, stride
                [3, 16, 16, True, 'RE', 1],  # NOTE: change stride 2 -> 1 for CIFAR10
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1280 if mode == 'large' else 1024

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nl1 = H_swish() # non-linearity
        self.layer = []
        in_planes = 16
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_planes, use_SE, NL, stride in self.cfg:
            self.layer.append(Block(in_planes, exp_size, out_planes, kernel_size, stride, use_SE, NL))
            in_planes = out_planes
        self.layers = nn.Sequential(*self.layer)
        out_planes = exp_size
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.nl2 = H_swish()
        self.conv3 = nn.Conv2d(out_planes, last_channels_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.nl3 = H_swish()
        self.linear = nn.Linear(last_channels_num, num_classes)


    def forward(self, x):
        out = self.nl1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.nl2(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = self.nl3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV3()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()