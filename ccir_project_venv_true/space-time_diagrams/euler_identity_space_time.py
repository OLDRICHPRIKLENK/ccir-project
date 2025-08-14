import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 1.0            # radius of circle in (x, y) plane
omega = 2*np.pi/5  # angular frequency
steps = 200
dt = 0.1
times = np.linspace(0, steps*dt, steps)

# Circular motion in x-y plane
x = R * np.cos(omega * times)
y = R * np.sin(omega * times)
t = times  # time axis for 3D plot

# --- Plot 3D helix ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(x, y, t, color='blue')
ax.set_xlabel("Real Space (x)")
ax.set_ylabel("Imaginary Plane (y)")
ax.set_zlabel("Time (t)")
ax.set_title("Helix: Circular Motion in (x, y) Plane over Time")
ax.view_init(elev=20, azim=30)

# --- Plot components ---
ax2 = fig.add_subplot(222)
ax2.plot(t, x, label="x(t) = cos", color='red')
ax2.plot(t, y, label="y(t) = sin", color='green')
ax2.set_xlabel("Time")
ax2.set_ylabel("Position")
ax2.set_title("Components vs Time")
ax2.legend()

ax3 = fig.add_subplot(224)
ax3.plot(t, x**2 + y**2, color='purple')
ax3.set_xlabel("Time")
ax3.set_ylabel("x² + y²")
ax3.set_title("Radius² (constant)")

plt.tight_layout()
plt.show()
