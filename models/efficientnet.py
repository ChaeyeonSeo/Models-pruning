import torch
import torch.nn as nn
import torch.nn.functional as F
import time

conv1_first_time = 0
bn1_first_time = 0
nl1_first_time = 0
conv1_time = 0
bn1_time = 0
nl1_time = 0
conv2_time = 0
bn2_time = 0
nl2_time = 0
conv3_time = 0
bn3_time = 0
se_time = 0
conv2_last_time = 0
bn2_last_time = 0
nl2_last_time = 0
avg_pool_time = 0
linear_time = 0

class SEModule(nn.Module):
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.SiLU(),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y


class Block(nn.Module):
    def __init__(self, in_planes, exp_factor, out_planes, kernel_size, stride):
        super(Block, self).__init__()

        self.exp_size = in_planes * exp_factor
        self.in_planes = in_planes
        self.stride = stride

        # Expansion
        self.conv1 = nn.Conv2d(in_planes, self.exp_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.exp_size)
        self.nl1 = nn.SiLU()  # non-linearity

        # Depthwise Convolution
        self.conv2 = nn.Conv2d(self.exp_size, self.exp_size, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, groups=self.exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(self.exp_size)
        self.nl2 = nn.SiLU()  # non-linearity

        self.se = nn.Sequential()  # SE module
        self.se = SEModule(self.exp_size)

        # Linear Pointwise Convolution
        self.conv3 = nn.Conv2d(self.exp_size, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, expand=False):
        global conv1_time, bn1_time, nl1_time, conv2_time, bn2_time, \
            nl2_time, se_time, conv3_time, bn3_time
        # Conv1
        start = time.time()
        out = self.conv1(x)
        conv1_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_time += (time.time() - start)
        start = time.time()
        out = self.nl1(out)
        nl1_time += (time.time() - start)
        start = time.time()
        # Conv2
        out = self.conv2(out)
        conv2_time += (time.time() - start)
        start = time.time()
        out = self.bn2(out)
        bn2_time += (time.time() - start)
        start = time.time()
        out = self.nl2(out)
        nl2_time += (time.time() - start)
        start = time.time()
        # SE
        out = self.se(out)
        se_time += (time.time() - start)
        start = time.time()
        # Conv3
        out = self.conv3(out)
        conv3_time += (time.time() - start)
        start = time.time()
        out = self.bn3(out)
        bn3_time += (time.time() - start)
        # Residual
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class EfficientNet(nn.Module):
    def __init__(self, mode='small', num_classes=10):
        super(EfficientNet, self).__init__()

        self.cfg = [
            # expansion, out_planes, num_blocks, kernel_size, stride
            [1, 16, 1, 3, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
            [6, 24, 2, 3, 1],
            [6, 40, 2, 5, 2],
            [6, 80, 3, 3, 2],
            [6, 112, 3, 5, 1],
            [6, 192, 4, 5, 2],
            [6, 320, 1, 3, 1]  # NOTE: change stride 2 -> 1 for CIFAR10
        ]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.nl1 = nn.SiLU()  # non-linearity
        self.layers = []
        in_planes = 32
        # Block
        layer = []
        for expansion, out_planes, num_blocks, kernel, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                # in_planes, exp_size, out_planes, kernel_size, stride
                layer.append(Block(in_planes, expansion, out_planes, kernel, stride))
                in_planes = out_planes
        self.layers = nn.Sequential(*layer)
        self.conv2 = nn.Conv2d(out_planes, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.nl2 = nn.SiLU()
        self.linear = nn.Linear(1280, num_classes)

        self.mode = 1

    def change_mode(self):
        self.mode = 2

    def forward(self, x):
        global conv1_first_time, bn1_first_time, nl1_first_time, conv1_time, \
            bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_time, conv3_time, \
            bn3_time, conv2_last_time, bn2_last_time, nl2_last_time, avg_pool_time, \
            linear_time

        # first
        start = time.time()
        out = self.conv1(x)
        conv1_first_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_first_time += (time.time() - start)
        start = time.time()
        out = self.nl1(out)
        nl1_first_time += (time.time() - start)
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
        out = self.nl2(out)
        nl2_last_time += (time.time() - start)
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
        return out, conv1_first_time, bn1_first_time, nl1_first_time, conv1_time, \
               bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_time, conv3_time, \
               bn3_time, conv2_last_time, bn2_last_time, nl2_last_time, avg_pool_time, \
               linear_time

def test():
    net = EfficientNet()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
