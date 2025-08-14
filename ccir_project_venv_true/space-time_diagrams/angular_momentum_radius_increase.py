import numpy as np
import matplotlib.pyplot as plt

# Parameters
m = 1.0
c = 1.0
omega = 2*np.pi/5  # fixed angular frequency
R0 = 1.0           # initial radius
alpha = 0.05       # radius growth rate per time unit
steps = 200
dt = 0.1
times = np.linspace(0, steps*dt, steps)

# Growing radius
R = R0 + alpha * times

# Circular motion in x-y plane with growing radius
x = R * np.cos(omega * times)
y = R * np.sin(omega * times)
t = times  # time axis

# Energy from transverse speed
v_perp = R * omega
momentum = m * v_perp  # natural units, non-relativistic approx
energy = np.sqrt((m*c**2)**2 + (momentum*c)**2)

# --- Plot 3D helix ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(x, y, t, color='blue')
ax.set_xlabel("Real Space (x)")
ax.set_ylabel("Imaginary Space (y)")
ax.set_zlabel("Time (t)")
ax.set_title("Expanding Helix (ω const, R increasing)")
ax.view_init(elev=20, azim=30)

# --- Energy vs Time ---
ax2 = fig.add_subplot(122)
ax2.plot(t, energy, color='red')
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("Total Energy E(t)")
ax2.set_title("Energy Increase from Radius Growth")
ax2.grid(True)

plt.tight_layout()
plt.show()
