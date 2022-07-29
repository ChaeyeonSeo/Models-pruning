# Models-pruning
1. Prune MobileNetv1,2,3, VGG16, and EfficientNet-b0 in various way using Torch-Pruning library (https://github.com/VainF/Torch-Pruning) 
2. Convert .pt to .ONNX
3. Quantize ONNX models
4. Deploy ONNX models on Rasberry Pi 3b+

- /checkpoints: .pt model files
- /deploy: Deployment code
  - deploy_onnx.py: Deploy tflite file on RPi
  
  ```python deploy_onnx_measurement.py --model [model name] --file [output_file_suffix ] ```

  - measure.py: Measure temperature and energy usage of RPi
  - metric_reader.py: Organize results of deployment
- /(model): Pruned .pt model files
- /models: MobileNetv1, MobileNetv2, and MobileNetv3 codes
  - (model).py: Layers are separated
  - (model)_default.py: Default model codes
- /onnx: Converted ONNX files
- draw_(graph).py: 
- measure_latency: Measure the latency of each layer
- pruning.py: Prune model files using Torch-Pruning library
  - You can choose model, pruning amount, fine-tuning epochs, layers to prune, and strategy using arguments
  -   ```python pruning.py --model [model name] --finetune_epochs [finetuning epochs] --prune [pruning amount] --layer [layers to prune] --strategy [pruning strategy]```

- pruning_pt.py: Prune model files given in class
- run.sh: Automatically run codes