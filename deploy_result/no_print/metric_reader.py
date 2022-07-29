import csv
import argparse

parser = argparse.ArgumentParser(description='Metric Printout')
parser.add_argument('--model', type=str, default='mobilenetv1', help='.csv will be appended to end')
parser.add_argument('--file', type=str, default='1', help='.csv will be appended to end')
args = parser.parse_args()
# model = args.model
file = args.file

SAMPLING_RATE = 1

models = [
            # 'mobilenetv1', 'mobilenetv1.uint8quant', 'mobilenetv1.uint8quant_static',
            #   'mobilenetv1_0.2', 'mobilenetv1_0.2.uint8quant', 'mobilenetv1_0.2.uint8quant_static',
            #   'mobilenetv2', 'mobilenetv2.uint8quant', 'mobilenetv2.uint8quant_static',
            #   'mobilenetv2_0.2', 'mobilenetv2_0.2.uint8quant', 'mobilenetv2_0.2.uint8quant_static',
            #   'mobilenetv3', 'mobilenetv3.uint8quant', 'mobilenetv3.uint8quant_static',
            #   'mobilenetv3_0.2', 'mobilenetv3_0.2.uint8quant', 'mobilenetv3_0.2.uint8quant_static',
              'efficientnet', 'efficientnet.uint8quant', 'efficientnet.uint8quant_static',
              'efficientnet_0.2', 'efficientnet_0.2.uint8quant', 'efficientnet_0.2.uint8quant_static',
              ]

time = []
inference = []
accuracy = []

folder = 'rpi_02_opset16'
out_fname = f'output_efficientnet_{folder}.csv'
header = "Accuracy, Latency, Energy,"
with open(out_fname, 'w') as out_file:
    out_file.write(header)
    out_file.write("\n")

for model in models:
    with open(out_fname, 'a+') as out_file:
        out_file.write(f"{model}\n")
    for file in [1, 2, 3]:
        with open(f'./{folder}/runtime_{model}_{file}.csv', 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i != 0:
                    time.append((row[0], row[1]))
                    inference.append(float(row[2]))
                    accuracy.append(float(row[-1]))

        energy = []
        with open(f'./{folder}/measurements_rpi_{model}_{file}.csv', 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i != 0:
                    energy.append(float(row[1]))

        avg_energy = sum(energy)
        avg_latency = sum(inference) / len(inference)
        # max_power = max(map(max, energy))

        with open(out_fname, 'a+') as out_file:
            out_file.write(f"{accuracy[-1]}, {inference[-1]:.4f}, {avg_energy:.4f}\n")
        # print(f'Accuracy:  {accuracy[0]} %')
        # print(f'Latency:   {avg_latency} ms')
        # # print(f'Max Power: {max_power} W')
        # print(f'Energy:    {avg_energy} mJ')

