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
from models.vgg16 import VGG16
from models.efficientnet import EfficientNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Training MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--model', type=str, default='mobilenetv3', help='mobilenetv1_default, mobilenetv2, or mobilenetv3')
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
    'mobilenetv1_default': MobileNet,
    'mobilenetv2': MobileNetV2,
    'mobilenetv3': MobileNetV3,
    'vgg16': VGG16,
    'efficientnet': EfficientNet
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

for epoch in range(num_epochs):
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
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    
    accuracy.loc[epoch, 'Training'] = 100. * train_correct / train_total
    total_time += (time.time() - start)
    # Testing phase
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
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1),100. * test_correct / test_total))
    accuracy.loc[epoch, 'Testing'] = 100. * test_correct / test_total
    if 100. * test_correct / test_total > max_acc:
        max_acc = 100. * test_correct / test_total
        torch.save(model, f"checkpoints/{model_name}_{epoch + 1}.pt")

torchsummary.summary(model, (3, 32, 32))
accuracy.to_csv(f'{model_name}/{model_name}_accuracy.csv')
print(f'Training time: {total_time}s')
