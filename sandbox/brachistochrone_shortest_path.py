import numpy as np
import matplotlib.pyplot as plt

# Given points A and B
xA, yA = 0.0, 0.0
xB, yB = 2.0, -2.0
g = 9.81

# Analytical solver for brachistochrone parameters R and theta_f
def brachistochrone_params(xA, yA, xB, yB):
    drop = yA - yB  # vertical drop (positive)
    
    def f(theta):
        R = drop / (1 - np.cos(theta))
        return R * (theta - np.sin(theta)) - (xB - xA)
    
    # Newton's method to solve for theta_f
    theta = np.pi / 2
    for _ in range(100):
        R = drop / (1 - np.cos(theta))
        f_val = R * (theta - np.sin(theta)) - (xB - xA)
        f_deriv = (drop * np.sin(theta) / (1 - np.cos(theta))**2) * (theta - np.sin(theta)) + R * (1 - np.cos(theta))
        theta -= f_val / f_deriv
    theta_f = theta
    R = drop / (1 - np.cos(theta_f))
    return R, theta_f

# Get R and theta_f
R, theta_f = brachistochrone_params(xA, yA, xB, yB)

# Parametric equations of the cycloid
theta_vals = np.linspace(0, theta_f, 300)
x_vals = xA + R * (theta_vals - np.sin(theta_vals))
y_vals = yA - R * (1 - np.cos(theta_vals))

# Analytical travel time
T_total = R * theta_f / np.sqrt(2 * g * R)

# Plot
plt.figure(figsize=(6, 4))
plt.plot(x_vals, y_vals, label='Brachistochrone (Cycloid)')
plt.scatter([xA, xB], [yA, yB], color='red', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Brachistochrone Curve\nTravel time: {T_total:.6f} s")
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# Print parameters
print(f"R = {R}")
print(f"theta_f = {theta_f}")
print(f"Travel time = {T_total:.6f} s")


