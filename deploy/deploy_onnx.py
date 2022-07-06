import numpy as np
import onnxruntime
import time
from tqdm import tqdm
import os
import argparse
from PIL import Image

start = time.time()
parser = argparse.ArgumentParser(description="EE379K - ONNX Runtime Enivronment script")
parser.add_argument('--model', type=str, default='mobilenetv1', help='prune val ex 05, 1, 5')
parser.add_argument('--device', type=str, default='mc1', help='rpi or mc1')
parser.add_argument('--file', type=str, default='1')
args = parser.parse_args()

model_name = args.model
file_name = args.file

model_path = './models/'
onnx_model_name = model_path + f"{model_name}.onnx"

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
# print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
# print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

total_time = 0
total = 0
right = 0
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("/home/student/HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("/home/student/HW3_files/test_deployment/", filename)).resize((32, 32)) as img:
        # print("Image shape:", np.float32(img).shape)

        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        
        # print("Input Image shape:", input_image.shape)

        # Run inference and get the prediction for the input image
        start_time = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        total_time += (time.time() - start_time)

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        true_class = filename.replace('.png', '').split('_')[-1]
        if pred_class == true_class:
            right += 1
        total += 1

end = time.time()
with open(f'runtime_{model_name}_{file_name}.csv', 'w') as file:
    file.write(f'start,end,inference,right,total,accuracy\n'
               f'{start},{end},{total_time},{right},{total},{100 * right/total}\n')
