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
import models.vgg16
import models.efficientnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Pruning MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of epochs to finetune')
parser.add_argument('--model', type=str, default='mobilenetv3', help='mobilenetv1, mobilenetv2, or mobilenetv3, efficientnet')
parser.add_argument('--prune', type=float, default=0.2)
parser.add_argument('--layer', type=str, default="one", help="one, two, three and all")
parser.add_argument('--mode', type=int, default=1, help="pruning: 1, measurement: 2")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--strategy', type=str, default="L1", help="L1, L2, and random")
args = parser.parse_args()

finetune_epochs = args.finetune_epochs
batch_size = args.batch_size
model_name = args.model
prune_val = args.prune
layer = args.layer
mode = args.mode
seed = args.seed
strategy_name = args.strategy

print('model: ', model_name, ' layer: ', layer, ' prune_val: ', prune_val, ' strategy: ', strategy_name)

# model name: mobilenetv1, mobilenetv2, mobilenetv3
# layer: all, one
# prune: 0.05 ~ 0.9
# finetune: 0 ~ 200
model_path = f"{model_name}/{layer}/{strategy_name}/prune_{prune_val}"

torch.manual_seed(seed)

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
    'mobilenetv1': models.mobilenetv1.MobileNet,
    'mobilenetv2': models.mobilenetv2.MobileNetV2,
    'mobilenetv3': models.mobilenetv3.MobileNetV3,
    'vgg16': models.vgg16.VGG16,
    'efficientnet': models.efficientnet.EfficientNet
}


def load_model(model, path=f"./checkpoints/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        # if print_msg:
            # print(f"[I] Model loaded from {path}")
            # j = 0
            # dct = {}
            # import pickle
            # for name, param in model.state_dict().items():
            #     if (("weight" in name) and ("conv2" in name)):
            #         dct[name] = param
            #         print('number: ', j)
            #         print(param)
            #         j += 1
            # with open('torch_pruning.pickle', 'wb') as handle:
            #     pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
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

# torchsummary.summary(model, (3, 32, 32))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# Training loop
max_acc = 0
accuracy = pd.DataFrame(index=range(finetune_epochs + 1), columns={'Testing'})


def train(model, epoch):
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
    #     if (batch_idx + 1) % 100 == 0:
    #         print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, finetune_epochs, batch_idx + 1,
    #                                                                          len(train_dataset) // batch_size,
    #                                                                          train_loss / (batch_idx + 1),
    #                                                                          100. * train_correct / train_total))
    # print('Accuracy of the model on the 60000 train images: % f %%' % (100. * train_correct / train_total))
    return loss


def test(model, epoch):
    global max_acc
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
    # print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), acc))
    if 100. * test_correct / test_total > max_acc:
        max_acc = 100. * test_correct / test_total
        torch.save(model, f"{model_path}/finetune_{epoch + 1}.pt")
    return acc


# load model
model = load_model(model)
test(model, 0)



# Pruning
model = model.to(torch.device('cpu'))
# 1. setup strategy
if strategy_name == 'L1':
    strategy = tp.strategy.L1Strategy()
elif strategy_name == 'L2':
    strategy = tp.strategy.L2Strategy()
else:
    strategy = tp.strategy.RandomStrategy()

# 2. build layer dependency for model
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))


def prune_conv(conv, amount):
    # 3. get a pruning plan from the dependency graph.
    pruning_index = strategy(conv.weight, amount=amount)
    pruning_plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
    pruning_plan.exec()


def prune_bn(bn, amount):
    pruning_index = strategy(bn.weight, amount=amount)
    pruning_plan = DG.get_pruning_plan(bn, tp.prune_batchnorm, pruning_index)
    pruning_plan.exec()


def prune_linear(linear, amount):
    pruning_index = strategy(linear.weight, amount=amount)
    pruning_plan = DG.get_pruning_plan(linear, tp.prune_linear, pruning_index)
    pruning_plan.exec()


if model_name == 'mobilenetv1':
    # first layer
    if layer == 'all':
        prune_conv(model.conv1, amount=prune_val)
        prune_bn(model.bn1, amount=prune_val)
        # i = 0
        for m in model.modules():
            if isinstance(m, models.mobilenetv1.Block):
                # print("number: ", i)
                prune_conv(m.conv1, amount=prune_val)
                prune_bn(m.bn1, amount=prune_val)
                prune_conv(m.conv2, amount=prune_val)
                prune_bn(m.bn2, amount=prune_val)
                # i += 1
    elif layer == 'one':
        i = 0
        for m in model.modules():
            if isinstance(m, models.mobilenetv1.Block):
                # print("number: ", i)
                prune_conv(m.conv2, amount=prune_val)
                i += 1

elif model_name == 'mobilenetv2':
    if layer == 'all':
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

elif model_name == 'mobilenetv3':  # mobilenetv3
    if layer == 'all':
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
    elif layer == 'three':
        for m in model.modules():
            if isinstance(m, models.mobilenetv3.Block):
                prune_conv(m.conv3, amount=prune_val)

elif model_name == 'vgg16':
    if layer == 'all': # Prune Conv, Linear1, and Linear2
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                prune_conv(m, amount=prune_val)
        prune_linear(model.linear1, amount=prune_val)
        prune_linear(model.linear2, amount=prune_val)
    elif layer == 'one': # Prune Conv
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                prune_conv(m, amount=prune_val)

else: # EfficientNet-b0
    if layer == 'all': # Conv1, bn1, Blocks(Conv1, bn1, Conv2,bn2, SE, Conv3, bn3), Conv2, bn2
        prune_conv(model.conv1, amount=prune_val)
        prune_bn(model.bn1, amount=prune_val)
        for m in model.modules():
            if isinstance(m, models.efficientnet.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_bn(m.bn1, amount=prune_val)
                prune_conv(m.conv2, amount=prune_val)
                prune_bn(m.bn2, amount=prune_val)
                prune_linear(m.se_linear1, amount=prune_val)
                prune_linear(m.se_linear2, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
                prune_bn(m.bn3, amount=prune_val)
        prune_conv(model.conv2, amount=prune_val)
        prune_bn(model.bn2, amount=prune_val)
    elif layer == 'one': # Blocks(Conv1)
        for m in model.modules():
            if isinstance(m, models.efficientnet.Block):
                prune_conv(m.conv1, amount=prune_val)
    elif layer == 'two': # Blocks(Conv1, Conv3)
        for m in model.modules():
            if isinstance(m, models.efficientnet.Block):
                prune_conv(m.conv1, amount=prune_val)
                prune_conv(m.conv3, amount=prune_val)
    elif layer == 'three': # Blocks (Conv3)
        for m in model.modules():
            if isinstance(m, models.efficientnet.Block):
                prune_conv(m.conv3, amount=prune_val)

max_acc = 0
model = model.to(torch.device(device))

print("model size: ", model_size(model))
# No fine-tuning after pruning
# torchsummary.summary(model, (3, 32, 32))
first_acc = test(model, 0)
accuracy.loc[0, 'Testing'] = first_acc
torch.save(model, f"{model_path}/finetune_0.pt")

# No fine-tuning accuracy
with open(f'{model_name}/{layer}/{strategy_name}/no_finetuning_accuracy.csv', 'a') as f:
    writer = csv.writer(f)
    data = [prune_val, first_acc]
    writer.writerow(data)

# Number of parameter
with open(f'{model_name}/{layer}/{strategy_name}/parameter_num.csv', 'a') as f1:
    writer1 = csv.writer(f1)
    parameter_num = model_size(model)
    data = [prune_val, parameter_num]
    writer1.writerow(data)

# Fine-tuning
for fine_tune_epoch in range(finetune_epochs):
    train(model, fine_tune_epoch)
    test(model, fine_tune_epoch)

# Fine-tuning accuracy
accuracy.to_csv(f'{model_path}/accuracy.csv', index=False)

# Fine-tuning best accuracy
with open(f'{model_name}/{layer}/{strategy_name}/finetuning_best_accuracy.csv', 'a') as f2:
    writer2 = csv.writer(f2)
    data = [prune_val, max_acc]
    writer2.writerow(data)

# random_input = torch.randn(1, 3, 32, 32).to(device)
# torch.onnx.export(model, random_input, f'checkpoints/{model_name}_{prune_val}_{finetune_epochs}.onnx', export_params=True, opset_version=10)
