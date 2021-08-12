import matplotlib.pyplot as plt

with open("./mlosstrain.txt") as f:

	err = [float(x.split()[2][0:6]) for x in f.read().split(",")]

print(min(err))

plt.figure()

plt.plot(err)

plt.savefig("loss.jpg")