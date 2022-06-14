from matplotlib import pyplot as plt

x1 = list(range(0,201))


plt.figure()
plt.title("Accuracy")

plt.plot(x1, y_01, label="Prune 0.1")
plt.plot(x1, y_02, label="Prune 0.2")
plt.plot(x1, y_03, label="Prune 0.3")

plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.grid()
plt.legend()

plt.savefig('./finetuning_accuracy.png')