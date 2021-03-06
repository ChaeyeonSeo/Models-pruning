from collections import deque

import numpy as np
import pandas as pd
import csv
import time
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import argparse

import torchsummary
import torch_pruning as tp

import models.mobilenetv1
import models.mobilenetv2
import models.mobilenetv3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Training MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--finetune_epochs', type=int, default=5, help='Number of epochs to finetune')
parser.add_argument('--model', type=str, default='mobilenetv1_default', help='mobilenetv1_default, mobilenetv2, or mobilenetv3')
parser.add_argument('--prune', type=float, default=0.05)
parser.add_argument('--layer', type=str, default="one", help="one, two, three and one")
parser.add_argument('--mode', type=int, default=1, help="pruning: 1, measurement: 2")
args = parser.parse_args()

finetune_epochs = args.finetune_epochs
batch_size = args.batch_size
model_name = args.model
prune_val = args.prune
layer = args.layer
mode = args.mode

# model name: mobilenetv1_default, mobilenetv2, mobilenetv3
# layer: one, one
# prune: 0.05 ~ 0.9
# finetune: 0 ~ 200
model_path = f"{model_name}/{layer}/repeat"

random_seed = 1
torch.manual_seed(random_seed)

input_size = 3 * 32 * 32
num_classes = 10

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_names = {
    'mobilenetv1_default': models.mobilenetv1.MobileNet,
    'mobilenetv2': models.mobilenetv2.MobileNetV2,
    'mobilenetv3': models.mobilenetv3.MobileNetV3,
}


# def load_model(model, path=f"./checkpoints/{model_name}.pt", print_msg=True):
def load_model(path=f"{model_path}/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        if print_msg:
            print(f"[I] Model loaded from {path}")
            # torchsummary.summary(model, (3, 32, 32))
        return model
    except:
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")


def model_size(model, count_zeros=True):
    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = np.sum(tensor.detach().cpu().numpy() != 0.0)

        total_params += t
        nonzero_params += nz
    if not count_zeros:
        return int(nonzero_params)
    else:
        return int(total_params)


model = model_names.get(model_name, models.mobilenetv1.MobileNet)()
model = model.to(torch.device(device))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# Training loop
iteration = 0
total_time = 0
max_acc = 0
accuracy = pd.DataFrame(index=range(finetune_epochs + 1), columns={'Testing'})


def train(model, epoch):
    global iteration
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(torch.device(device))
        labels = labels.to(torch.device(device))

        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # if (batch_idx + 1) % 100 == 0:
        #     print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, finetune_epochs, batch_idx + 1,
        #                                                                      len(train_dataset) // batch_size,
        #                                                                      train_loss / (batch_idx + 1),
        #                                                                      100. * train_correct / train_total))
        iteration += 1
    # print('Accuracy of the model on the 60000 train images: % f %%' % (100. * train_correct / train_total))
    return loss


def test(model, epoch, iteration=0):
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(torch.device(device))
            labels = labels.to(torch.device(device))
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    acc = 100. * test_correct / test_total
    accuracy.loc[epoch + 1, 'Testing'] = acc
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), acc))
    # if 100. * test_correct / test_total > max_acc_in:
    #     max_acc_in = 100. * test_correct / test_total
    #     print("max: %.2f" % max_acc_in)
    return acc

# load model
model = load_model()
# torchsummary.summary(model, (3, 32, 32))


def prune_conv(conv, amount):
    strategy = tp.strategy.L1Strategy()
    # 3. get a pruning plan from the dependency graph.
    pruning_index = strategy(conv.weight, amount=amount)
    pruning_plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
    pruning_plan.exec()


def prune_bn(bn, amount):
    strategy = tp.strategy.L1Strategy()
    pruning_index = strategy(bn.weight, amount=amount)
    plan = DG.get_pruning_plan(bn, tp.prune_batchnorm, pruning_index)
    plan.exec()


model = model.to(torch.device(device))

# torchsummary.summary(model, (3, 32, 32))
first_acc = test(model, 0)
accuracy.loc[0, 'Testing'] = first_acc
iteration_in = 1

# Pruning
# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy()

base = test(model, 0)
params = deque(2*[0], 2)
real_model_name = model_name
model_name = model_path + '/' + model_name +'.pt'
# model_name = f'{model_path}/{real_model_name}_{prune_val}.pt'
params = deque(2*[0], 2)
while base > 70:
    print("============================= iteration: %d =============================" % iteration_in)
    model = load_model(model_name)
    print("Before pruning: ", model_size(model))
    model = model.to(torch.device('cpu'))
    params.appendleft(model_size(model))
    if abs(params[0] - params[1]) < 1000:
        prune_val += .01
        prune_val = round(prune_val, 2)
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))
    if real_model_name == 'mobilenetv1_default':
        for m in model.modules():
            if isinstance(m, models.mobilenetv1.Block):
                prune_conv(m.conv2, amount=prune_val)
    elif real_model_name == 'mobilenetv2':
        for m in model.modules():
            if isinstance(m, models.mobilenetv2.Block):
                prune_conv(m.conv1, amount=prune_val)
    else:  # mobilenetv3
        for m in model.modules():
            if isinstance(m, models.mobilenetv3.Block):
                prune_conv(m.conv1, amount=prune_val)
    model = model.to(torch.device(device))
    print("After pruning: ", model_size(model))
    # torchsummary.summary(model, (3, 32, 32))
    criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters())

    base = 0
    for fine_tune_epoch in range(finetune_epochs):
        train(model, fine_tune_epoch)
        acc = test(model, fine_tune_epoch)
        if acc > base:
            base = acc
            if acc > 70:
                print("acc:", acc)
                torch.save(model, f'{model_path}/{real_model_name}_{prune_val}.pt')
    model_name = f'{model_path}/{real_model_name}_{prune_val}.pt'
    acc = test(model, iteration_in)
    # write a row to the csv file
    with open(f'{model_path}/iteration_max_accuracy.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        data = [base]
        writer.writerow(data)

    # write a row to the csv file
    with open(f'{model_path}/iteration_parameter_num.csv', 'a') as f1:
        # create the csv writer
        writer1 = csv.writer(f1)
        parameter_num = model_size(model)
        data = [parameter_num]
        writer1.writerow(data)

    print('iteration complete')
    iteration_in += 1



torch.save(model, f"{model_path}/max.pt")
'''
if model_name == 'mobilenetv1_default':
    # first layer
    if layer == "one":
        prune_conv(model.conv1, amount=prune_val)
        prune_bn(model.bn1, amount=prune_val)
        for m in model.modules():
            if isinstance(m, models.mobilenetv1_default.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_bn(m.bn1, amount=prune_val)
                prune_conv(m.conv2, amount=prune_val)
                prune_bn(m.bn2, amount=prune_val)
        # prune_linear(model.linear, amount=prune_val)
    else:
        for m in model.modules():
            if isinstance(m, models.mobilenetv1_default.Block):
                prune_conv(m.conv2, amount=prune_val)

elif model_name == 'mobilenetv2':
    if layer =='one':
        prune_conv(model.conv1, amount=prune_val)
        prune_bn(model.bn1, amount=prune_val)
        for m in model.modules():
            if isinstance(m, models.mobilenetv2.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_bn(m.bn1, amount=prune_val)
                prune_conv(m.conv2, amount=prune_val)
                prune_bn(m.bn2, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
                prune_bn(m.bn3, amount=prune_val)
        prune_conv(model.conv2, amount=prune_val)
        prune_bn(model.bn2, amount=prune_val)
        # prune_linear(model.linear, amount=prune_val)
    elif layer == 'one':
        for m in model.modules():
            if isinstance(m, models.mobilenetv2.Block):
                prune_conv(m.conv1, amount=prune_val)
    elif layer == 'two':
        for m in model.modules():
            if isinstance(m, models.mobilenetv2.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
    elif layer == 'three':
        for m in model.modules():
            if isinstance(m, models.mobilenetv2.Block):
                prune_conv(m.conv3, amount=prune_val)

else: # mobilenetv3
    if layer == 'one':
        prune_conv(model.conv1, amount=prune_val)
        prune_bn(model.bn1, amount=prune_val)
        for m in model.modules():
            if isinstance(m, models.mobilenetv3.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_bn(m.bn1, amount=prune_val)
                prune_conv(m.conv2, amount=prune_val)
                prune_bn(m.bn2, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
                prune_bn(m.bn3, amount=prune_val)
        prune_conv(model.conv2, amount=prune_val)
        prune_bn(model.bn2, amount=prune_val)
        prune_conv(model.conv3, amount=prune_val)
    if layer == 'one':
        for m in model.modules():
            if isinstance(m, models.mobilenetv3.Block):
                prune_conv(m.conv1, amount=prune_val)
    elif layer == 'two':
        for m in model.modules():
            if isinstance(m, models.mobilenetv3.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
'''

# # Fine-tuning
# for fine_tune_epoch in range(finetune_epochs):
#     train(model, fine_tune_epoch)
#     test(model, fine_tune_epoch)


accuracy.to_csv(f'{model_path}/accuracy.csv', index=False)
# torchsummary.summary(model, (3, 32, 32))
# test(model, 0)

# # open the file in the write mode
# with open(f'{model_path}/finetuning_best_accuracy.csv', 'a') as f:
#     # create the csv writer
#     writer = csv.writer(f)
#     # write a row to the csv file
#     data = [prune_val, max_acc]
#     writer.writerow(data)


# random_input = torch.randn(1, 3, 32, 32).to(device)
# torch.save(model, f'checkpoints/{model_name}_{prune_val}_all_layer.pt')
# torch.onnx.export(model, random_input, f'checkpoints/{model_name}_{prune_val}_{finetune_epochs}.onnx', export_params=True, opset_version=10)
