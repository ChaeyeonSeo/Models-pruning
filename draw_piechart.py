import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Conv1', 'bn1', 'ReLU1', 'Conv2', 'bn2', 'ReLU2'
sizes = [0.02256488800048828, 0.028546571731567383, 0.009987831115722656, 0.08763718605041504, 0.033600568771362305, 0.011137247085571289]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('mobilenetv1.png')
plt.show()

