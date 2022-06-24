import torch
import torch.nn as nn
import torch.nn.functional as F
import time

pooling_time = 0
conv_time = 0
relu_time = 0
linear1_time = 0
relu1_time = 0
linear2_time = 0
relu2_time = 0
linear3_time = 0

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, case=0):
        super(Block, self).__init__()
        self.case = case
        if case == 0:
            self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

    def forward(self, x):
        global conv_time, pooling_time
        if self.case == 0:
            start = time.time()
            out = self.pooling(x)
            pooling_time += (time.time() - start)
            return out
        start = time.time()
        out = self.conv(x)
        conv_time += (time.time() - start)
        start = time.time()
        out = self.relu(out)
        return out

class VGG16(nn.Module):
    def __init__(self, vgg_name="VGG16", mode=1):
        super(VGG16, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)
        self.mode = mode

    def change_mode(self):
        self.mode = 2

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            if x == 'M':
                layers.append(Block(in_planes, out_planes, 0))
            else:
                layers.append(Block(in_planes, out_planes, 1))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        global linear1_time, relu1_time, linear2_time, relu2_time, linear3_time
        out = self.features(x)
        out = out.view(out.size(0), -1)
        start = time.time()
        out = self.linear1(out)
        linear1_time += (time.time() - start)
        start = time.time()
        out = self.relu1(out)
        relu1_time += (time.time() - start)
        start = time.time()
        out = self.linear2(out)
        linear2_time += (time.time() - start)
        start = time.time()
        out = self.relu2(out)
        relu2_time += (time.time() - start)
        start = time.time()
        out = self.linear3(out)
        linear3_time += (time.time() - start)

        # Pruning
        if self.mode == 1:
            return out

        return out, conv_time, relu_time, pooling_time, linear1_time, relu1_time, linear2_time, relu2_time, linear3_time


def test():
    net = VGG16()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    for m in net.modules():
        print(m)

# test()