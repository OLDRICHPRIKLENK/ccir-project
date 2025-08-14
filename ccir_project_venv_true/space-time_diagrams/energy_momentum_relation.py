import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 1.0
c = 1.0
dt = 0.05
steps = 50

# Acceleration setup: exponential momentum growth
p0 = 0.1
momentum = [p0]
for i in range(1, steps+1):
    momentum.append(momentum[-1] * 2)  # double momentum each step

momentum = np.array(momentum)
times = np.linspace(0, steps*dt, steps+1)

# Compute velocity from momentum (relativistic)
gamma = np.sqrt(1 + (momentum/(m*c))**2)  # gamma from p = gamma m v
velocity = momentum / (gamma * m)

# Compute position over time
x = np.zeros_like(times)
for i in range(1, len(times)):
    x[i] = x[i-1] + velocity[i-1] * dt

# Compute total relativistic energy
energy = np.sqrt((m*c**2)**2 + (momentum*c)**2)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# Space-time diagram
axs[0].plot(x, times, marker='o')
axs[0].set_xlabel("Space (x)")
axs[0].set_ylabel("Time (t)")
axs[0].set_title("Space–Time Diagram (Exponential Acceleration)")
axs[0].invert_yaxis()  # time upwards like Minkowski diagram

# Energy-time diagram
axs[1].plot(times, energy, marker='o', color='red')
axs[1].set_xlabel("Time (t)")
axs[1].set_ylabel("Total Energy E(t)")
axs[1].set_title("Energy vs Time (Exponential Momentum Increase)")
axs[1].grid(True)

plt.tight_layout()
plt.show()
