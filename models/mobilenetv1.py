'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

total_time = 0
conv1_first_time = 0
conv1_time = 0
bn1_time = 0
relu1_time = 0
conv2_time = 0
bn2_time = 0
relu2_time = 0
avg_pool_time = 0
linear_time = 0

class Block(nn.Module):
    global total_time, conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes=3, out_planes=32, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        global total_time, conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time
        start = time.time()
        out = self.conv1(x)
        conv1_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_time += (time.time() - start)
        start = time.time()
        out = F.relu(out)
        relu1_time += (time.time() - start)
        start = time.time()
        out = self.conv2(out)
        conv2_time += (time.time() - start)
        start = time.time()
        out = self.bn2(out)
        bn2_time += (time.time() - start)
        start = time.time()
        out = F.relu(out)
        relu2_time += (time.time() - start)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, mode=1):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.mode = mode

    def change_mode(self):
        # mode 1: pruning
        # mode 2: measurement
        self.mode = 2

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if (x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        global conv1_first_time, relu1_time, bn1_time, avg_pool_time, linear_time
        start = time.time()
        out = self.conv1(x)
        conv1_first_time += (time.time() - start)
        start = time.time()
        out = self.bn1(out)
        bn1_time += (time.time() - start)
        start = time.time()
        out = F.relu(out)
        relu1_time += (time.time() - start)
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        start = time.time()
        out = F.avg_pool2d(out, 2)
        avg_pool_time += (time.time() - start)
        out = out.view(out.size(0), -1)
        start = time.time()
        out = self.linear(out)
        linear_time += (time.time() - start)

        # Pruning
        if self.mode == 1:
            return out

        # Measurement
        return out, conv1_first_time, conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, avg_pool_time, linear_time

def test():
    net = torch.load
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())


# test()