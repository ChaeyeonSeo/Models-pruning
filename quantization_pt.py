import argparse
import numpy as np
from PIL import Image
import os
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader
from pathlib import Path


def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data


class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, 32, 32, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input.1': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EE379K - ONNX Qunatizer")
    parser.add_argument('--model', type=str, default='efficientnet', help='prune val ex 05, 1, 5')
    parser.add_argument('--pruning', type=str, default='0.2', help='prune val ex 05, 1, 5')
    args = parser.parse_args()

    model_name = args.model
    pruning_amount = args.pruning
    model = f'onnx/{model_name}_{pruning_amount}.onnx'
    model_quant = f'onnx/dynamic_quantized/{model_name}_{pruning_amount}.uint8quant.onnx'
    model_quant_static = f'onnx/static_quantized/{model_name}_{pruning_amount}.uint8quant_static.onnx'
    # model = f'onnx/{model_name}.onnx'
    # model_quant = f'onnx/dynamic_quantized/{model_name}.uint8quant.onnx'
    # model_quant_static = f'onnx/static_quantized/{model_name}.uint8quant_static.onnx'
    quantize_dynamic(Path(model), Path(model_quant), weight_type=QuantType.QUInt8)
    dr = MobilenetDataReader('onnx/Project/test_deployment')
    quantize_static(model, model_quant_static, dr)

