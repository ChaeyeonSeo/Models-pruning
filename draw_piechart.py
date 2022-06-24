import matplotlib.pyplot as plt
import numpy

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Linear1', 'Linear2', 'Linear3', 'Conv'
sizes = numpy.array([262656, 262656, 5130, 14714688])


def absolute_value(val):
    # a = numpy.round(val / 100. * sizes.sum(), 0)
    a  = sizes[ numpy.abs(sizes - val/100.*sizes.sum()).argmin() ]
    return a


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct=absolute_value,
        startangle=170)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Number of parameters')
plt.savefig('vgg16_parameters_num.png')
plt.show()
