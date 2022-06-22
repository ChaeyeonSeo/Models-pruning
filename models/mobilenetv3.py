'''MobileNetV3 in PyTorch.

See the paper "Searching for MobileNetV3"
for more details.
'''
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
se_avg_time = 0
se_linear1_time = 0
se_nl1_time = 0
se_linear2_time = 0
se_nl2_time = 0
se_mult_time = 0
conv2_last_time = 0
bn2_last_time = 0
nl2_last_time = 0
avg_pool_time = 0
conv3_last_time = 0
nl3_last_time = 0
linear_time = 0


class H_sigmoid(nn.Module):
    def __init__(self):
        super(H_sigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class H_swish(nn.Module):
    def __init__(self):
        super(H_swish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class Block(nn.Module):
    def __init__(self, in_planes, exp_size, out_planes, kernel_size, stride, use_SE, NL):
        super(Block, self).__init__()

        use_HS = NL == 'HS'

        self.exp_size = exp_size
        self.in_planes = in_planes
        self.stride = stride
        self.reduction_ratio = 4
        self.use_SE = use_SE

        # Expansion
        self.conv1 = nn.Conv2d(in_planes, exp_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=exp_size)
        self.nl1 = nn.ReLU()  # non-linearity
        if use_HS:
            self.nl1 = H_swish()

        # Depthwise Convolution
        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, groups=exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.nl2 = nn.ReLU()  # non-linearity
        if use_HS:
            self.nl2 = H_swish()

        # Squeeze-and-Excite
        if use_SE:
            self.se_avg_pool = nn.AdaptiveAvgPool2d(1)
            self.se_linear1 = nn.Linear(exp_size, exp_size // self.reduction_ratio, bias=False)
            self.se_nl1 = nn.ReLU()
            self.se_linear2 = nn.Linear(exp_size // self.reduction_ratio, exp_size, bias=False)
            self.se_nl2 = H_sigmoid()

        # Linear Pointwise Convolution
        self.conv3 = nn.Conv2d(exp_size, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, expand=False):
        global conv1_time, bn1_time, nl1_time, conv2_time, bn2_time, \
            nl2_time, se_avg_time, se_linear1_time, se_nl1_time, \
            se_linear2_time, se_nl2_time, se_mult_time, conv3_time, bn3_time
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
        # Conv2
        start = time.time()
        out = self.conv2(out)
        conv2_time += (time.time() - start)
        start = time.time()
        out = self.bn2(out)
        bn2_time += (time.time() - start)
        start = time.time()
        out = self.nl2(out)
        nl2_time += (time.time() - start)
        # SE
        if self.use_SE:
            batch_size, channel_num, _, _ = out.size()
            start = time.time()
            out_se = self.se_avg_pool(out).view(batch_size, channel_num)
            se_avg_time += (time.time() - start)
            start = time.time()
            out_se = self.se_linear1(out_se)
            se_linear1_time += (time.time() - start)
            start = time.time()
            out_se = self.se_nl1(out_se)
            se_nl1_time += (time.time() - start)
            start = time.time()
            out_se = self.se_linear2(out_se)
            se_linear2_time += (time.time() - start)
            start = time.time()
            out_se = self.se_nl2(out_se)
            se_nl2_time += (time.time() - start)
            out_se = out_se.view(batch_size, channel_num, 1, 1)
            start = time.time()
            out = out*out_se
            se_mult_time += (time.time() - start)
        # Conv3
        start = time.time()
        out = self.conv3(out)
        conv3_time += (time.time() - start)
        start = time.time()
        out = self.bn3(out)
        bn3_time += (time.time() - start)
        # Residual
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, case='small', num_classes=10):
        super(MobileNetV3, self).__init__()

        if case == 'large':  # MobileNetV3-Large
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
        elif case == 'small':  # MobileNetV3-Small
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

        last_channels_num = 1280 if case == 'large' else 1024

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nl1 = H_swish()  # non-linearity
        self.layer = []
        in_planes = 16
        # Blocks
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

        self.mode = 1

    def change_mode(self):
        self.mode = 2

    def forward(self, x):
        global conv1_first_time, bn1_first_time, nl1_first_time, conv1_time, \
            bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_avg_time, \
            se_linear1_time, se_nl1_time, se_linear2_time, se_nl2_time, \
            se_mult_time, conv3_time, bn3_time, conv2_last_time, bn2_last_time, \
            nl2_last_time, avg_pool_time, conv3_last_time, nl3_last_time, linear_time

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
        # Avg Pooling
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        start = time.time()
        out = F.avg_pool2d(out, 4)
        avg_pool_time += (time.time() - start)
        # 1x1 Conv
        start = time.time()
        out = self.conv3(out)
        conv3_last_time += (time.time() - start)
        start = time.time()
        out = self.nl3(out)
        nl3_last_time += (time.time() - start)
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
            bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_avg_time, \
            se_linear1_time, se_nl1_time, se_linear2_time, se_nl2_time, se_mult_time, \
            conv3_time, bn3_time, conv2_last_time, bn2_last_time, nl2_last_time, \
            avg_pool_time, conv3_last_time, nl3_last_time, linear_time


def test():
    net = MobileNetV3()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
