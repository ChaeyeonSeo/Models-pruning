import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchsummary
import argparse

from mobilenet_rm_filt_pt import MobileNet, remove_channel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Pruning MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--finetune_epochs', type=int, default=0, help='Number of epochs to finetune')
parser.add_argument('--model', type=str, default='mobilenetv1', help='mobilenetv1, mobilenetv2, or mobilenetv3')
parser.add_argument('--prune', type=float, default=0.0)
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

pruning_method = "chn_prune"
# pruning_method = "mag_prune"
fine_tune = True

print('model: ', model_name, ' layer: ', layer, ' prune_val: ', prune_val, ' strategy: ', strategy_name)

model_path = f"{model_name}/code/prune_{prune_val}"

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


def load_model(model, path="./checkpoints/mobilenetv1.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        if print_msg:
            print(f"[I] Model loaded from {path}")
            # j = 0
            # dct = {}
            # import pickle
            # for name, param in model.state_dict().items():
            #     if (("weight" in name) and ("conv2" in name)):
            #         dct[name] = param
            #         print('number: ', j)
            #         print(param)
            #         j += 1
            # with open('code_pruning.pickle', 'wb') as handle:
            #     pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return model
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
    # except:
    #     if print_msg:
    #         print(f"[E] Model failed to be loaded from {path}")


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

model = MobileNet()
model.load_state_dict(torch.load('./checkpoints/mobilenetv1.pt').state_dict())
model = model.to(torch.device('cuda'))

criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())
iteration = 0

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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), acc))
    if 100. * test_correct / test_total > max_acc:
        max_acc = 100. * test_correct / test_total
        torch.save(model, f"{model_path}/finetune_{epoch + 1}.pt")
    return acc


# if load_my_model:
#     load_model(model)
#     test(0, mode='Non-pruned model', value='True')
# else:
#     for epoch in range(num_epochs):
#         train(epoch)
#         test(epoch, mode='test', value='True')
#     exit(0)

# model = load_model(model)
test(model, 0)

def prune_model_thres(model, threshold):
    mask_dict = {}
    for name, param in model.state_dict().items():
        if ("weight" in name):
            param[param.abs() < threshold] = 0
            mask_dict[name] = torch.where(torch.abs(param) > 0, 1, 0)
    model.mask_dict = mask_dict


def channel_fraction_pruning(model, fraction=0.2):
    mask_dict = {}
    number = 0
    for name, param in model.state_dict().items():
        if (("weight" in name) and ("conv2" in name)):
            # print('number: ', number)
            # print(param)
            score_list = torch.sum(torch.abs(param), dim=(1, 2, 3)).to('cpu')
            # print(score_list)
            removed_idx = []
            threshold = np.percentile(np.abs(score_list), fraction * 100)
            # print('threshold: ', threshold)
            for i, score in enumerate(score_list):
                # print('i: ', i, 'score: ', score)
                if score < threshold:
                    removed_idx.append(i)
                param[removed_idx, :, :, :] = 0
                mask_dict[name] = torch.where(torch.abs(param) > 0, 1, 0)
            # print(len(removed_idx))
            # print(removed_idx)
            number += 1
    model.mask_dict = mask_dict
    # print(mask_dict)


res1 = []
num_params = []
# for prune_thres in np.arange(0, 1, 0.1):
#     print('Before pruning:', model_size(model))
#     if pruning_method == 'mag_prune':
#         prune_model_thres(model, prune_thres)
#     else:
#         channel_fraction_pruning(model, prune_thres)
#     model._apply_mask()
#     model = remove_channel(model)
#     print('After pruning:', model_size(model))
#     model = model.to(device)
#     # if fine_tune:
#     #     for fine_tune_epoch in range(1):
#     #         train(fine_tune_epoch)
#     #         train(fine_tune_epoch)
#     acc = test(1, "thres", prune_thres)
#     res1.append([prune_thres, acc])
#     num_params.append(model_size(model, prune_thres == 0))

# print('Before pruning:', model_size(model))
# print(model.layers[11].conv1.weight)
if pruning_method == 'mag_prune':
    prune_model_thres(model, prune_val)
else:
    channel_fraction_pruning(model, prune_val)
model._apply_mask()
model = remove_channel(model)
# print('After pruning:', model_size(model))
print(model.layers[11].conv1.weight)
model = model.to(device)
# torchsummary.summary(model, (3, 32, 32))
max_acc = 0

print("model size: ", model_size(model))

first_acc = test(model, 0)
accuracy.loc[0, 'Testing'] = first_acc
torch.save(model, f"{model_path}/finetune_0.pt")

# No fine-tuning accuracy
with open(f'{model_name}/code/no_finetuning_accuracy.csv', 'a') as f:
    writer = csv.writer(f)
    data = [prune_val, first_acc]
    writer.writerow(data)

# Number of parameter
with open(f'{model_name}/code/parameter_num.csv', 'a') as f1:
    writer1 = csv.writer(f1)
    parameter_num = model_size(model)
    data = [prune_val, parameter_num]
    writer1.writerow(data)
    
for fine_tune_epoch in range(finetune_epochs):
    train(model, fine_tune_epoch)
    test(model, fine_tune_epoch)

accuracy.to_csv(f'{model_path}/accuracy.csv', index=False)

# Fine-tuning best accuracy
with open(f'{model_name}/code/finetuning_best_accuracy.csv', 'a') as f2:
    writer2 = csv.writer(f2)
    data = [prune_val, max_acc]
    writer2.writerow(data)

# plt.figure()
# plt.plot(num_params, res1[:, 1])
# plt.title('{}: vs #Params'.format(pruning_method))
# plt.xlabel('Number of parameters')
# plt.ylabel('Test accuracy')
# plt.savefig('{}_param_fine_tune_{}.png'.format(pruning_method, fine_tune))
# plt.close()
#
# plt.figure()
# plt.plot(np.arange(0, 1, 0.02), res1[:, 1])
# plt.title('{}: Accuracy vs Threshold'.format(pruning_method))
# plt.xlabel('Pruning Threshold')
# plt.ylabel('Test accuracy')
# plt.savefig('{}_thresh_fine_tune_{}.png'.format(pruning_method, fine_tune))
# plt.close()
