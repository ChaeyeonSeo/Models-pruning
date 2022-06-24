from matplotlib import pyplot as plt

x1 = []


plt.figure()
plt.title("Parameter vs. Accuracy")

plt.plot(x1, y1, label="MobileNet V1")
plt.plot(x2, y2, label="MobileNet V2")
plt.plot(x3, y3, label="MobileNet V3")

plt.xlabel("Number of Parameter")
plt.ylabel("Accuracy [%]")
plt.grid()
plt.legend()

plt.savefig('./vgg_parameter_accuracy.png')