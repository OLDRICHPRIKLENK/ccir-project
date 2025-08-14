import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
m = 1.0
c = 1.0
dt = 0.05
steps = 50

# Momentum growth (exponential)
p0 = 0.1
momentum = [p0]
for i in range(1, steps+1):
    momentum.append(momentum[-1] * 2)
momentum = np.array(momentum)
times = np.linspace(0, steps*dt, steps+1)

# Velocity from momentum (relativistic)
gamma = np.sqrt(1 + (momentum/(m*c))**2)
velocity = momentum / (gamma * m)

# Position over time
x = np.zeros_like(times)
for i in range(1, len(times)):
    x[i] = x[i-1] + velocity[i-1] * dt

# Relativistic total energy
energy = np.sqrt((m*c**2)**2 + (momentum*c)**2)

# 3D plot
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, times, energy, marker='o', color='blue')

ax.set_xlabel("Space (x)")
ax.set_ylabel("Time (t)")
ax.set_zlabel("Energy E(t)")
ax.set_title("3D Space–Time–Energy Diagram (Exponential Momentum Increase)")

ax.view_init(elev=25, azim=45)
plt.show()
