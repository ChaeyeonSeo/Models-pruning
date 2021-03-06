import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, exp_factor, out_planes, kernel_size, stride):
        super(Block, self).__init__()

        self.exp_size = in_planes * exp_factor
        self.in_planes = in_planes
        self.stride = stride
        self.reduction_ratio = 4

        # Expansion
        self.conv1 = nn.Conv2d(in_planes, self.exp_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.exp_size)
        self.nl1 = nn.SiLU()  # non-linearity

        # Depthwise Convolution
        self.conv2 = nn.Conv2d(self.exp_size, self.exp_size, kernel_size=kernel_size, stride=stride,
                               padding=(kernel_size - 1) // 2, groups=self.exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(self.exp_size)
        self.nl2 = nn.SiLU()  # non-linearity

        # Squeeze-and-Excite
        self.se_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se_linear1 = nn.Linear(self.exp_size, self.exp_size // self.reduction_ratio, bias=False)
        self.se_nl1 = nn.SiLU()
        self.se_linear2 = nn.Linear(self.exp_size // self.reduction_ratio, self.exp_size, bias=False)
        self.se_nl2 = nn.Sigmoid()

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
        # Conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nl1(out)
        # Conv2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nl2(out)
        # SE
        batch_size, channel_num, _, _ = out.size()
        out_se = self.se_avg_pool(out).view(batch_size, channel_num)
        out_se = self.se_linear1(out_se)
        out_se = self.se_nl1(out_se)
        out_se = self.se_linear2(out_se)
        out_se = self.se_nl2(out_se)
        out_se = out_se.view(batch_size, channel_num, 1, 1)
        out = out * out_se
        # Conv3
        out = self.conv3(out)
        out = self.bn3(out)
        # Residual
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes=10):
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
        out_planes = 0
        # Block
        layer = []
        for expansion, out_planes, num_blocks, kernel, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layer.append(Block(in_planes, expansion, out_planes, kernel, stride))
                in_planes = out_planes
        self.layers = nn.Sequential(*layer)
        self.conv2 = nn.Conv2d(out_planes, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.nl2 = nn.SiLU()
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        # first
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nl1(out)
        # blocks
        out = self.layers(out)
        # 1x1 Conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nl2(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # Linear
        out = self.linear(out)

        return out

def test():
    net = EfficientNet()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
