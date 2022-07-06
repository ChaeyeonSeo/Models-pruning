import argparse

import torch

import models.mobilenetv1
import models.mobilenetv2
import models.mobilenetv3
import models.vgg16
import models.efficientnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Argument parser
parser = argparse.ArgumentParser(description='Pruning MobileNet V1, V2, and V3')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--finetune_epochs', type=int, default=100, help='Number of epochs to finetune')
parser.add_argument('--model', type=str, default='efficientnet', help='mobilenetv1, mobilenetv2, mobilenetv3, or efficientnet')
parser.add_argument('--prune', type=float, default=0.2)
parser.add_argument('--layer', type=str, default="one", help="one, two, three and all")
parser.add_argument('--mode', type=int, default=1, help="pruning: 1, measurement: 2")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--strategy', type=str, default="L1", help="L1, L2, and random")
args = parser.parse_args()

finetune_epochs = args.finetune_epochs
batch_size = args.batch_size
model_name = args.model
prune_val = args.prune
layer = args.layer
mode = args.mode
seed = args.seed
strategy_name = args.strategy

print('model: ', model_name, ' layer: ', layer, ' prune_val: ', prune_val, ' strategy: ', strategy_name)
# print(torch.__version__)
# model name: mobilenetv1, mobilenetv2, mobilenetv3
# layer: all, one
# prune: 0.05 ~ 0.9
# finetune: 0 ~ 200
model_path = f"{model_name}/{layer}/{strategy_name}/prune_{prune_val}"

model_names = {
    'mobilenetv1': models.mobilenetv1.MobileNet,
    'mobilenetv2': models.mobilenetv2.MobileNetV2,
    'mobilenetv3': models.mobilenetv3.MobileNetV3,
    'vgg16': models.vgg16.VGG16,
    'efficientnet': models.efficientnet.EfficientNet
}

model = model_names.get(model_name, models.mobilenetv1.MobileNet)()
model = model.to(torch.device(device))


def load_model(model, path=f"{model_path}/{model_name}.pt", print_msg=True):
# def load_model(model, path=f"checkpoints/{model_name}.pt", print_msg=True):
    try:
        model = torch.load(path, map_location=torch.device(device))
        return model
    # except:
    #     if print_msg:
    #         print(f"[E] Model failed to be loaded from {path}")
    except Exception as e:
        print(e)

model = load_model(model)

random_input = torch.randn(1, 3, 32, 32).to(device)
# torch.onnx.export(model, random_input, f'./onnx/{model_name}.onnx', export_params=True, opset_version=16)
torch.onnx.export(model, random_input, f'./onnx/{model_name}_{prune_val}.onnx', export_params=True, opset_version=16)
