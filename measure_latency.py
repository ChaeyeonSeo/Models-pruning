import csv
import sys
import torch
import torch.nn as nn
import torchsummary
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import time
import pandas as pd

from models.mobilenetv1 import MobileNet
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Training MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--model', type=str, default='mobilenetv2', help='mobilenetv1, mobilenetv2, or mobilenetv3')
parser.add_argument('--prune', type=float, default=0.1)
parser.add_argument('--layer', type=str, default="one", help="all, one, two, and three")
parser.add_argument('--mode', type=int, default=2, help="pruning: 1, measurement: 2")
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
batch_size = args.batch_size
model_name = args.model
prune_val = args.prune
layer = args.layer
mode = args.mode

model_path = f"{model_name}/{layer}/prune_{prune_val}"

random_seed = 1
torch.manual_seed(random_seed)

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
    'mobilenetv1': MobileNet,
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
}

model = model_names.get(model_name, MobileNet)()
model = model.to(torch.device(device))

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# Training loop
total_time = 0
conv1_first_time = 0
bn1_first_time = 0
relu1_first_time = 0
nl1_first_time = 0
conv1_time = 0
bn1_time = 0
relu1_time = 0
nl1_time = 0
conv2_time = 0
bn2_time = 0
relu2_time = 0
nl2_time = 0
conv3_time = 0
bn3_time = 0
se_time = 0
conv2_last_time = 0
bn2_last_time = 0
nl2_last_time = 0
relu2_last_time = 0
avg_pool_time = 0
conv3_last_time = 0
nl3_last_time = 0
linear_time = 0


def load_model(model, path=f"{model_path}/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        model.change_mode()
        if print_msg:
            print(f"[I] Model loaded from {path}")
        return model
    except:
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")


def test(model, epoch):
    global total_time, conv1_first_time, bn1_first_time, relu1_first_time, conv1_time, \
        bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time, \
        conv2_last_time, bn2_last_time, relu2_last_time, avg_pool_time, linear_time

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
            start_total = time.time()
            if model_name == 'mobilenetv1':
                outputs, conv1_first_time, conv1_time, bn1_time, relu1_time, \
                conv2_time, bn2_time, relu2_time, avg_pool_time, linear_time = model(images)

            elif model_name == 'mobilenetv2':
                outputs, conv1_first_time, bn1_first_time, relu1_first_time, conv1_time, \
                bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time, \
                conv2_last_time, bn2_last_time, relu2_last_time, avg_pool_time, linear_time = model(images)

            else:  # mobilenetv3
                outputs, conv1_first_time, bn1_first_time, nl1_first_time, conv1_time, \
                bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_time, conv3_time, \
                bn3_time, conv2_last_time, bn2_last_time, nl2_last_time, avg_pool_time, \
                conv3_last_time, linear_time = model(images)

            total_time += (time.time() - start_total)
            # print(outputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), 100. * test_correct / test_total))
    # print(conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time)


model = load_model(model)
test(model, 0)

# print(conv1_time, bn1_time, relu1_time, conv2_time, bn2_time, relu2_time)
if model_name == 'mobilenetv1':
    layer_labels = ['Conv_first', 'Conv1', 'bn1', 'ReLU1', 'Conv2', 'bn2',
                    'ReLU2', 'Pooling', 'Linear']
    sizes = [conv1_first_time, conv1_time, bn1_time, relu1_time, conv2_time,
             bn2_time, relu2_time, avg_pool_time, linear_time]

elif model_name == 'mobilenetv2':
    layer_labels = ['Conv_first', 'bn_first', 'relu_first', 'Conv1',
                    'bn1', 'ReLU1', 'Conv2', 'bn2', 'ReLU2', 'Conv3', 'bn3',
                    'Conv_last', 'bn_last', 'relu_last', 'Pooling', 'Linear']
    sizes = [conv1_first_time, bn1_first_time, relu1_first_time, conv1_time,
             bn1_time, relu1_time, conv2_time, bn2_time, relu2_time, conv3_time, bn3_time,
             conv2_last_time, bn2_last_time, relu2_last_time, avg_pool_time, linear_time]

else:  # mobilenetv3
    layer_labels = ['Conv1_first', 'bn1_first', 'nl1_first', 'Conv1',
                    'bn1', 'nl1', 'Conv2', 'bn2', 'nl2', 'SE', 'Conv3', 'bn3',
                    'Conv2_last', 'bn2_last', 'nl2_last', 'Pooling',
                    'Conv3_last', 'bn3_last', 'Linear']
    sizes = [conv1_first_time, bn1_first_time, nl1_first_time, conv1_time,
             bn1_time, nl1_time, conv2_time, bn2_time, nl2_time, se_time, conv3_time,
             bn3_time, conv2_last_time, bn2_last_time, nl2_last_time, avg_pool_time,
             conv3_last_time, nl3_last_time, linear_time]

df = pd.DataFrame(data={'layer': layer_labels, 'value': sizes})
df = df.sort_values('value', ascending=False)

# df2 = df[:9].copy()
#
# new_row = pd.DataFrame(data={
#     'layer': ['others'],
#     'value': [df['value'][9:].sum()]
# })
#
# df2 = pd.concat([df2, new_row])

fig1, ax1 = plt.subplots()
# ax1.pie(df2['value'], labels=df2['layer'], autopct='%1.1f%%', startangle=180)
ax1.pie(sizes, labels=layer_labels, autopct='%1.1f%%', startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.savefig(f'{model_path}/layers_{prune_val}.png')
# plt.show()

# torchsummary.summary(model, (3, 32, 32))
print(f'Total time: {total_time}s')

# open the file in the write mode
with open(f'{model_name}/{layer}/inference_time.csv', 'a') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv files
    data = [total_time]
    writer.writerow(data)
