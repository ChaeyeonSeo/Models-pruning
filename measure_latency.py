import sys
import torch
import torch.nn as nn
import torchsummary
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import time
import pandas as pd

from models.mobilenetv1 import MobileNet
from models.mobilenetv2 import MobileNetV2
from models.mobilenetv3 import MobileNetV3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Training MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--model', type=str, default='mobilenetv1', help='mobilenetv1, mobilenetv2, or mobilenetv3')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
model_name = args.model

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
max_acc = 0
accuracy = pd.DataFrame(index=range(num_epochs), columns={'Training', 'Testing'})

def load_model(model, path=f"./checkpoints/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        if print_msg:
            print(f"[I] Model loaded from {path}")
        return model
    except:
        if print_msg:
            print(f"[E] Model failed to be loaded from {path}")

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


model = load_model(model)
test(model, 0, mode='Non-pruned model', value='True')

torchsummary.summary(model, (3, 32, 32))
print(f'Training time: {total_time}s')
