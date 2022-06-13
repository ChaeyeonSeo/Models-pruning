'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

total_time = 0
conv1_first_time = 0
bn1_first_time = 0
relu1_first_time = 0
conv1_time = 0
bn1_time = 0
relu1_time = 0
conv2_time = 0
bn2_time = 0
relu2_time = 0
conv3_time = 0
bn3_time = 0
conv2_last_time = 0
bn2_last_time = 0
relu2_last_time = 0
avg_pool_time = 0
linear_time = 0

class Block(nn.Module):
    '''expand + depthwise + one'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        global conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time

        # Expand
        start = time.time()
        out = self.conv1(x)
        conv1_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_time += (time.time() - start)
        start = time.time()
        out = F.relu6(out)
        relu1_time += (time.time() - start)
        # Depthwise Conv
        start = time.time()
        out = self.conv2(out)
        conv2_time += (time.time() - start)
        start = time.time()
        out = self.bn2(out)
        bn2_time += (time.time() - start)
        start = time.time()
        out = F.relu6(out)
        relu2_time += (time.time() - start)
        # Pointwise Conv
        start = time.time()
        out = self.conv3(out)
        conv3_time += (time.time() - start)
        start = time.time()
        out = self.bn3(out)
        bn3_time += (time.time() - start)
        # Residual
        # shortcut time?
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        self.mode = 1

    def change_mode(self):
        self.mode = 2

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        global conv1_first_time, bn1_first_time, relu1_first_time, conv1_time, \
            bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time, \
            conv2_last_time, bn2_last_time, relu2_last_time, avg_pool_time, linear_time

        # first
        start = time.time()
        out = self.conv1(x)
        conv1_first_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_first_time += (time.time() - start)
        start = time.time()
        out = F.relu6(out)
        relu1_first_time += (time.time() - start)
        # blocks
        out = self.layers(out)
        # 1x1 Conv
        start = time.time()
        out = self.conv2(out)
        conv2_last_time += (time.time() - start)
        start = time.time()
        out = self.bn2(out)
        bn2_last_time += (time.time() - start)
        start = time.time()
        out = F.relu6(out)
        relu2_last_time += (time.time() - start)
        # Avg Pooling
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        start = time.time()
        out = F.avg_pool2d(out, 4)
        avg_pool_time += (time.time() - start)
        out = out.view(out.size(0), -1)
        # Linear
        start = time.time()
        out = self.linear(out)
        linear_time += (time.time() - start)

        # Pruning
        if self.mode == 1:
            return out

        # Measurement
        return out, conv1_first_time, bn1_first_time, relu1_first_time, conv1_time, \
            bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time, \
            conv2_last_time, bn2_last_time, relu2_last_time, avg_pool_time, linear_time


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()