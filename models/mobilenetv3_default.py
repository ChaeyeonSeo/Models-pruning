'''MobileNetV3 in PyTorch.

See the paper "Searching for MobileNetV3"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SEModule(nn.Module):
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y


class Block(nn.Module):
    def __init__(self, in_planes, exp_size, out_planes, kernel_size, stride, use_SE, NL):
        super(Block, self).__init__()

        use_HS = NL == 'HS'

        self.exp_size = exp_size
        self.in_planes = in_planes
        self.stride = stride

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
        self.se = nn.Sequential()  # SE module
        if use_SE:
            self.se = SEModule(exp_size)
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
        out = self.nl1(self.bn1(self.conv1(x)))
        out = self.nl2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):
    def __init__(self, mode='small', num_classes=10):
        super(MobileNetV3, self).__init__()

        if mode == 'large': # MobileNetV3-Large
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
        elif mode == 'small': # MobileNetV3-Small
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

        last_channels_num = 1280 if mode == 'large' else 1024

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
