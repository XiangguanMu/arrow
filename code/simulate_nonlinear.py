import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
# y=sin(sin x)
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(np.sin(x))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y=sin(sin x)')
plt.savefig('results/sin_sin_x.png')
# plt.show()