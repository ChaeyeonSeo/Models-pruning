import torch
import torch.nn as nn

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, case=0):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.case = case

    def forward(self, x):
        if self.case == 0:
            out = self.pooling(x)

        out = self.conv(x)

class VGG16(nn.Module):
    def __init__(self, vgg_name="VGG16"):
        super(VGG16, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linear1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 10)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_planes = 3
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(Block(in_planes, out_planes, 1))
                in_planes = out_planes
        return nn.Sequential(*layers)

def test():
    net = VGG16()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    for m in net.modules():
        print(m)

# test()