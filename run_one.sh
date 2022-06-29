#!/bin/sh

#python pruning.py --prune 0.0
#python pruning.py --prune 0.05
#python pruning.py --prune 0.1
#python pruning.py --prune 0.2
#python pruning.py --prune 0.3
#python pruning.py --prune 0.4
#python pruning.py --prune 0.5
# Error -> delete +1 in channel number
python pruning.py --prune 0.6
python pruning.py --prune 0.7
python pruning.py --prune 0.8
python pruning.py --prune 0.9
## L2 MobileNet V1
#python pruning.py --model mobilenetv1_default --prune 0.0 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.05 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.1 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.2 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.3 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.4 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.5 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.6 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.7 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.8 --layer one --strategy L2
#python pruning.py --model mobilenetv1_default --prune 0.9 --layer one --strategy L2
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
#
## L2 Efficientnet
#python pruning.py --model efficientnet --prune 0.0 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.05 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.1 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.2 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.3 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.4 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.5 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.6 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.7 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.8 --layer one --strategy L2
#python pruning.py --model efficientnet --prune 0.9 --layer one --strategy L2

#python pruning.py --prune 0.05 --finetune_epochs 200
#python pruning.py --prune 0.1 --finetune_epochs 200 --layer "one"
#python pruning.py --prune 0.2 --finetune_epochs 200 --layer "one"
#python pruning.py --prune 0.3 --finetune_epochs 200 --layer "one"
