#!/bin/sh

python pruning_pt.py --prune 0.0
python pruning_pt.py --prune 0.05
python pruning_pt.py --prune 0.1
python pruning_pt.py --prune 0.2
python pruning_pt.py --prune 0.3
python pruning_pt.py --prune 0.4
python pruning_pt.py --prune 0.5
python pruning_pt.py --prune 0.6
python pruning_pt.py --prune 0.7
python pruning_pt.py --prune 0.8
python pruning_pt.py --prune 0.9


## vgg L2
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.0 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.05 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.1 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.2 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.3 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.4 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.5 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.6 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.7 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.8 --layer one --strategy L1
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.9 --layer one --strategy L1
#
##python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.0 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.05 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.1 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.2 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.3 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.4 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.5 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.6 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.7 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.8 --layer one --strategy L1
#python pruning.py --finetune_epochs 10 --model vgg16 --prune 0.9 --layer one --strategy L1
#
##python pruning.py --prune 0.0 --layer two
##python pruning.py --prune 0.05 --layer two
##python pruning.py --prune 0.1 --layer two
##python pruning.py --prune 0.2 --layer two
##python pruning.py --prune 0.3 --layer two
##python pruning.py --prune 0.4 --layer two
##python pruning.py --prune 0.5 --layer two
##python pruning.py --prune 0.6 --layer two
##python pruning.py --prune 0.7 --layer two
##python pruning.py --prune 0.8 --layer two
##python pruning.py --prune 0.9 --layer two
##
##python pruning.py --prune 0.0 --layer three
##python pruning.py --prune 0.05 --layer three
##python pruning.py --prune 0.1 --layer three
##python pruning.py --prune 0.2 --layer three
##python pruning.py --prune 0.3 --layer three
##python pruning.py --prune 0.4 --layer three
##python pruning.py --prune 0.5 --layer three
##python pruning.py --prune 0.6 --layer three
##python pruning.py --prune 0.7 --layer three
##python pruning.py --prune 0.8 --layer three
##python pruning.py --prune 0.9 --layer three
#
##python measure_latency.py --prune 0.0
##python measure_latency.py --prune 0.05
##python measure_latency.py --prune 0.1
##python measure_latency.py --prune 0.2
##python measure_latency.py --prune 0.3
##python measure_latency.py --prune 0.4
##python measure_latency.py --prune 0.5
##python measure_latency.py --prune 0.6
##python measure_latency.py --prune 0.7
##python measure_latency.py --prune 0.8
##python measure_latency.py --prune 0.9
#
## python pruning.py --prune 0.05 --finetune_epochs 200
##python pruning.py --prune 0.1 --finetune_epochs 200
##python pruning.py --prune 0.2 --finetune_epochs 200
##python pruning_mobilenet.py --prune 0.3 --finetune_epochs 200
