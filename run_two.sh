#!/bin/sh
# mobilenetv2 two
python pruning_mobilenet.py --prune 0.0 --layer two
python pruning_mobilenet.py --prune 0.05 --layer two
python pruning_mobilenet.py --prune 0.1 --layer two
python pruning_mobilenet.py --prune 0.2 --layer two
python pruning_mobilenet.py --prune 0.3 --layer two
python pruning_mobilenet.py --prune 0.4 --layer two
python pruning_mobilenet.py --prune 0.5 --layer two
python pruning_mobilenet.py --prune 0.6 --layer two
python pruning_mobilenet.py --prune 0.7 --layer two
python pruning_mobilenet.py --prune 0.8 --layer two
python pruning_mobilenet.py --prune 0.9 --layer two

#python measure_latency.py --prune 0.0
#python measure_latency.py --prune 0.05
#python measure_latency.py --prune 0.1
#python measure_latency.py --prune 0.2
#python measure_latency.py --prune 0.3
#python measure_latency.py --prune 0.4
#python measure_latency.py --prune 0.5
#python measure_latency.py --prune 0.6
#python measure_latency.py --prune 0.7
#python measure_latency.py --prune 0.8
#python measure_latency.py --prune 0.9

#python pruning_mobilenet.py --prune 0.05 --finetune_epochs 200
#python pruning_mobilenet.py --prune 0.1 --finetune_epochs 200 --layer "all"
#python pruning_mobilenet.py --prune 0.2 --finetune_epochs 200 --layer "all"
#python pruning_mobilenet.py --prune 0.3 --finetune_epochs 200 --layer "all"
