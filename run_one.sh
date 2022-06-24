#!/bin/sh
## L2 MobileNet V1
#python pruning.py --model mobilenetv1 --prune 0.0 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.05 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.1 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.2 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.3 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.4 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.5 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.6 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.7 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.8 --layer one --strategy L2
#python pruning.py --model mobilenetv1 --prune 0.9 --layer one --strategy L2
#
## L2 MobileNet V2
#python pruning.py --model mobilenetv2 --prune 0.0 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.05 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.1 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.2 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.3 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.4 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.5 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.6 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.7 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.8 --layer one --strategy L2
#python pruning.py --model mobilenetv2 --prune 0.9 --layer one --strategy L2
#
## L2 MobileNet V3
#python pruning.py --model mobilenetv3 --prune 0.0 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.05 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.1 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.2 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.3 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.4 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.5 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.6 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.7 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.8 --layer one --strategy L2
#python pruning.py --model mobilenetv3 --prune 0.9 --layer one --strategy L2
#
## L2 VGG
#python pruning.py --model vgg16 --prune 0.0 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.05 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.1 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.2 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.3 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.4 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.5 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.6 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.7 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.8 --layer one --strategy L2
#python pruning.py --model vgg16 --prune 0.9 --layer one --strategy L2

# L2 Efficientnet
python pruning.py --model efficientnet --prune 0.0 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.05 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.1 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.2 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.3 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.4 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.5 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.6 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.7 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.8 --layer one --strategy L2
python pruning.py --model efficientnet --prune 0.9 --layer one --strategy L2

#python pruning.py --prune 0.05 --finetune_epochs 200
#python pruning.py --prune 0.1 --finetune_epochs 200 --layer "all"
#python pruning.py --prune 0.2 --finetune_epochs 200 --layer "all"
#python pruning.py --prune 0.3 --finetune_epochs 200 --layer "all"
