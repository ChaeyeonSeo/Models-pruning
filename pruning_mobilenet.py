import numpy as np
import pandas as pd
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
# from models.mobilenetv1 import MobileNet, Block
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Training MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--model', type=str, default='mobilenetv3', help='mobilenetv1, mobilenetv2, or mobilenetv3')
parser.add_argument('--prune', type=float, default=0.1)
parser.add_argument('--finetune', type=bool, default=False)
args = parser.parse_args()

num_epochs = args.epochs
batch_size = args.batch_size
model_name = args.model
load_my_model = True
prune_val = args.prune
fine_tuning = args.finetune

random_seed = 1
torch.manual_seed(random_seed)

input_size = 3*32*32
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
    'mobilenetv1': models.mobilenetv1.MobileNet,
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
}

def load_model(model, path=f"./checkpoints/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        if print_msg:
            print(f"[I] Model loaded from {path}")
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
accuracy = pd.DataFrame(index=range(num_epochs), columns={'Training', 'Testing'})

def train(model, epoch):
    global iteration, total_time
    start = time.time()
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
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
        iteration += 1
    accuracy.loc[epoch, 'Training'] = 100. * train_correct / train_total
    total_time += (time.time() - start)
    print('Accuracy of the model on the 60000 train images: % f %%' % (100*accuracy))
    return loss

def test(model, epoch, mode, value):
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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), 100. * test_correct / test_total))
    accuracy.loc[epoch, 'Testing'] = 100. * test_correct / test_total
    if 100. * test_correct / test_total > max_acc:
        max_acc = 100. * test_correct / test_total
        torch.save(model, f"checkpoints/{model_name}_{epoch + 1}.pt")

if load_my_model:
    model = load_model(model)
    test(model, 0, mode='Non-pruned model', value='True')
else:
    for epoch in range(num_epochs):
        train(epoch)
        test(epoch, mode='test', value='True')
        torch.save(model, f'./checkpoints/{model_name}.pt')
    exit(0)

torchsummary.summary(model, (3, 32, 32))

model = model.to(torch.device('cpu'))
# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L1Strategy()

# 2. build layer dependency for model
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,32,32))

def prune_conv(conv, amount):
    strategy = tp.strategy.L1Strategy()
    # 3. get a pruning plan from the dependency graph.
    pruning_index = strategy(conv.weight, amount=amount)
    pruning_plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
    pruning_plan.exec()

block = 'models.'+{model_name}+'.Block'
# first layer
prune_conv(model.conv1, amount=prune_val)
for m in model.modules():
    if isinstance(m, block):
        prune_conv(m.conv1, amount=prune_val)
        prune_conv(m.conv2, amount=prune_val)

# if fine_tuning:
#     for fine_tune_epoch in range(num_epochs):
#         train(model, fine_tune_epoch)
#     acc = test(model, 1, "thres", prune_val)
#     with open('accuracy.txt', 'a') as f:
#         f.write(f'Prune: {prune_val}, Accuracy: {acc}%\n')

model = model.to(torch.device(device))
torchsummary.summary(model, (3, 32, 32))
accuracy.to_csv(f'{model_name}_accuracy.csv')

random_input = torch.randn(1, 3, 32, 32).to(device)

torch.save(model, f'checkpoints/{model_name}_{prune_val}.pt')
torch.onnx.export(model, random_input, f'checkpoints/{model_name}_{prune_val}_{num_epochs}.onnx', export_params=True, opset_version=10)
