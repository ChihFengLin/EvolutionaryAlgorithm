import numpy as np
import math
import matplotlib.pyplot as plt

x2 = np.linspace(0, math.sqrt(10), 2000)
x1 = np.sqrt(10-(x2**2))
f1 = x1
f2 = (x2)**3


plt.figure(1)
plt.plot(f1, f2)
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("Pareto Front")

plt.figure(2)
plt.plot(x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("X-Y Space")
plt.show()
