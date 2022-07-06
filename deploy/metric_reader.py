import csv
import argparse
from deploy.measurement import SAMPLING_RATE

parser = argparse.ArgumentParser(description='Metric Printout')
parser.add_argument('--file', type=str, default='log', help='.csv will be appended to end')
args = parser.parse_args()
file = args.file

time = []
inference = []
accuracy = []
with open(f'{file}_run/runtime_{file}_mc1.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i != 0:
            time.append((row[0], row[1]))
            inference.append(float(row[2]))
            accuracy.append(float(row[-1]))

energy = [[], [], []]
with open(f'{file}_run/measurements_mc1_{file}.csv', 'r') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i != 0:
            if (float(time[0][0]) - 2 * SAMPLING_RATE) < float(row[0]) < (float(time[0][1]) + 2 * SAMPLING_RATE):
                energy[0].append(float(row[1]))
            elif (float(time[1][0]) - 2 * SAMPLING_RATE) < float(row[0]) < (float(time[1][1]) + 2 * SAMPLING_RATE):
                energy[1].append(float(row[1]))
            elif (float(time[2][0]) - 2 * SAMPLING_RATE) < float(row[0]) < (float(time[2][1]) + 2 * SAMPLING_RATE):
                energy[2].append(float(row[1]))

avg_energy = sum(map(sum, energy)) * SAMPLING_RATE / 3
avg_latency = sum(inference) / len(inference)
max_power = max(map(max, energy))

print(f'Accuracy:  {accuracy[0]} %')
print(f'Latency:   {avg_latency} ms')
print(f'Max Power: {max_power} W')
print(f'Energy:    {avg_energy} mJ')

