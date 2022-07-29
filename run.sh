#!/bin/sh

# Default
python deploy_onnx_measurement.py --model mobilenetv1 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet --file 3
pkill python
sleep 5m

# Pruned
python deploy_onnx_measurement.py --model mobilenetv1_0.2 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2_0.2 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3_0.2 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2 --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet_0.2 --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2 --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2 --file 3
pkill python
sleep 5m

# Dynamic quantized Default
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet.uint8quant --file 3
pkill python
sleep 5m

# Dynamic quantized Pruned
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant --file 3
pkill python
sleep 5m

# Static quantized Default
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet.uint8quant_static --file 3
pkill python
sleep 5m

# Static quantized Pruned
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv1_0.2.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv2_0.2.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model mobilenetv3_0.2.uint8quant_static --file 3
pkill python
sleep 5m

python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant_static --file 1
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant_static --file 2
pkill python
sleep 5m
python deploy_onnx_measurement.py --model effcientnet_0.2.uint8quant_static --file 3
pkill python
sleep 5m
