"""
    MobileNet-v1 model written in PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
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
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

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
            stride = 1 if isinstance(x, int) else x[1]
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


    def _apply_mask(self):
        # print(self.mask_dict.keys())
        for name, param in self.state_dict().items():
            if name in self.mask_dict.keys():
                # print(param.data.shape, self.mask_dict[name].shape)
                param.data *= self.mask_dict[name]
            #
            # else:
            #     print("name: ", name)


def remove_channel(input_model):
    '''
    Input: model
           description: the pruned model
    Ouput: new_model
           description: the new model generating by removing one-zero channels
    '''

    new_model = copy.deepcopy(input_model)
    score_list = torch.sum(torch.abs(new_model.conv1.weight.data), dim=(1,2,3))
    next_layer_score_list = torch.sum(torch.abs(new_model.layers[0].conv1.weight.data), dim=(1,2,3))
    score_list = score_list * next_layer_score_list
    out_planes_num = int(torch.count_nonzero(score_list))
    out_planes_idx = torch.squeeze( torch.nonzero(score_list, as_tuple=False))
    conv1_wgt=copy.deepcopy(new_model.conv1.weight.data)
    new_model.conv1 = nn.Conv2d(3, out_planes_num, kernel_size=3, stride=1, padding=1, bias=False)
    new_model.bn1 = nn.BatchNorm2d(out_planes_num)
    new_model.conv1.weight.data[:,:,:,:] = conv1_wgt[out_planes_idx,:,:,:]

    in_planes_num = out_planes_num
    in_planes_idx = out_planes_idx
    for i, block in enumerate(new_model.layers):
        if i in [1, 3 ,5, 11]:
            stride = 2
        else:
            stride = 1
        conv1_wgt=copy.deepcopy(block.conv1.weight.data)
        new_model.layers[i].conv1 = nn.Conv2d(in_planes_num, in_planes_num, kernel_size=3, stride=stride,
                                              padding=1, groups=in_planes_num, bias=False)
        new_model.layers[i].bn1 =  nn.BatchNorm2d(in_planes_num)
        new_model.layers[i].conv1.weight.data[:,:,:,:] = conv1_wgt[in_planes_idx,:,:,:]
        score_list = torch.sum(torch.abs(block.conv2.weight.data), dim=(1,2,3))
        if i <len(new_model.layers)-1:
            next_layer_score_list = torch.sum(torch.abs(new_model.layers[i+1].conv1.weight.data), dim=(1,2,3))
            score_list = score_list * next_layer_score_list
        out_planes_num = int(torch.count_nonzero(score_list))
        # print("out planes num: ", out_planes_num)
        out_planes_idx = torch.squeeze(torch.nonzero(score_list, as_tuple=False))
        conv2_wgt=copy.deepcopy(block.conv2.weight.data)
        new_model.layers[i].conv2 = nn.Conv2d(in_planes_num, out_planes_num, kernel_size=1, stride=1,
                                              padding=0, bias=False)
        new_model.layers[i].bn2 = nn.BatchNorm2d(out_planes_num)

        for idx_out,n in enumerate(out_planes_idx):
            new_model.layers[i].conv2.weight.data[idx_out,:,:,:] = conv2_wgt[n,in_planes_idx,:,:]
        in_planes_num = out_planes_num
        in_planes_idx = out_planes_idx
    lin_wgt=copy.deepcopy(new_model.linear.weight.data)
    lin_bias=copy.deepcopy(new_model.linear.bias.data)
    new_model.linear = nn.Linear(in_planes_num, 10)

    new_model.linear.weight.data = lin_wgt[:,out_planes_idx]
    new_model.linear.bias.data = lin_bias
    return new_model
