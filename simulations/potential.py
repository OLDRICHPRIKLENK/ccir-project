import numpy as np
import matplotlib.pyplot as plt

# ------------------
# Trap & ion parameters
# ------------------
Vrf = 200.0        # V
Omega = 2.0e7      # rad/s
r0 = 1.5e-3        # m
z0 = 3.0e-3        # m
kappa = 0.30
Uec = 5.0          # V

# Ca+ properties
m = 6.64e-26       # kg
e_charge = 1.602176634e-19  # C

# ------------------
# Potential functions
# ------------------
def phi_linear(x, y, z, t):
    """Instantaneous potential for a linear Paul trap."""
    rf_part = (Vrf*np.cos(Omega*t)/(2*r0**2)) * (x**2 - y**2)
    dc_part = (kappa*Uec/z0**2) * z**2
    return rf_part + dc_part  # volts

def grad_rf_potential(x, y, z):
    """Gradient of RF spatial potential (without cos term)."""
    coeff = Vrf/(2*r0**2)
    dVdx = coeff * 2*x
    dVdy = coeff * -2*y
    dVdz = 0.0
    return np.array([dVdx, dVdy, dVdz])

def pseudo_potential(x, y, z):
    gradV = grad_rf_potential(x, y, z)
    return (e_charge**2 / (4*m*Omega**2)) * np.dot(gradV, gradV)  # Joules

# ------------------
# Grid for plotting
# ------------------
grid_size = 40
extent = 1.5e-3  # ± range in x,y,z (m)

x_vals = np.linspace(-extent, extent, grid_size)
y_vals = np.linspace(-extent, extent, grid_size)
z_vals = np.linspace(-extent, extent, grid_size)

X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

# Choose plot type
plot_mode = "instant"  # "instant" or "pseudo"
t_fixed = 0.0         # time for instant plot

if plot_mode == "instant":
    Phi = phi_linear(X, Y, Z, t_fixed)
elif plot_mode == "pseudo":
    # convert to eV for readability
    Phi = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                Phi[i,j,k] = pseudo_potential(X[i,j,k], Y[i,j,k], Z[i,j,k]) / e_charge
else:
    raise ValueError("plot_mode must be 'instant' or 'pseudo'")

# ------------------
# 3D surface plot (slice at z=0 plane)
# ------------------
mid_idx = grid_size // 2
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X[:,:,mid_idx]*1e3, Y[:,:,mid_idx]*1e3, Phi[:,:,mid_idx],
                       cmap='viridis', edgecolor='none')

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("Potential (V)" if plot_mode=="instant" else "Pseudo-potential (eV)")
ax.set_title(f"Linear Paul Trap — {plot_mode} potential (z=0 plane)")
fig.colorbar(surf, shrink=0.5, aspect=10, label=ax.zaxis.label.get_text())

plt.tight_layout()
plt.show()
