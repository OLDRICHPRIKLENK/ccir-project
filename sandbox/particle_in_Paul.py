import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------
# Trap parameters (linear trap)
# ------------------
Vrf = 200.0        # RF amplitude (V)
Omega = 2.0e7      # RF angular frequency (rad/s)
r0 = 1.5e-3        # radial scale (m)
z0 = 3.0e-3        # axial scale to endcaps (m)
kappa = 0.30       # geometry factor
Uec = 5.0          # endcap DC voltage (V)

# Ion parameters (Ca+)
m = 6.64e-26       # kg
e_charge = 1.602176634e-19  # C

# ------------------
# Potential and gradient
# ------------------
def phi_linear(x, y, z, t):
    rf_part = (Vrf*np.cos(Omega*t)/(2*r0**2)) * (x**2 - y**2)
    dc_part = (kappa*Uec/z0**2) * z**2
    return rf_part + dc_part  # volts

def grad_phi_linear(x, y, z, t):
    gx = (Vrf*np.cos(Omega*t)/(r0**2)) * x
    gy = -(Vrf*np.cos(Omega*t)/(r0**2)) * y
    gz = (2*kappa*Uec/z0**2) * z
    return gx, gy, gz  # volts/m

# ------------------
# Euler–Lagrange equations → ODE system
# ------------------
def rhs(t, Y):
    x, y, z, vx, vy, vz = Y
    gx, gy, gz = grad_phi_linear(x, y, z, t)
    ax = (-e_charge * gx) / m
    ay = (-e_charge * gy) / m
    az = (-e_charge * gz) / m
    return [vx, vy, vz, ax, ay, az]

# ------------------
# Initial conditions (position m, velocity m/s)
# ------------------
x0, y0, z0_ion = 0.5e-3, 0.0, 0.0   # 0.5 mm offset in x
vx0, vy0, vz0 = 0.0, 0.0, 0.0
Y0 = [x0, y0, z0_ion, vx0, vy0, vz0]

# ------------------
# Integrate motion
# ------------------
n_rf_periods = 200
steps_per_period = 500
T_rf = 2*np.pi / Omega
t_end = n_rf_periods * T_rf
t_eval = np.linspace(0, t_end, int(n_rf_periods*steps_per_period) + 1)

sol = solve_ivp(rhs, (0, t_end), Y0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-11)
t = sol.t
x, y, z, vx, vy, vz = sol.y

# ------------------
# 3D trajectory plot
# ------------------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x*1e3, y*1e3, z*1e3, lw=1)
ax.scatter([0], [0], [0], color='red', s=50, label="Trap center")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_zlabel("z (mm)")
ax.set_title("Ion trajectory in linear Paul trap")
ax.legend()
plt.tight_layout()
plt.show()

# ------------------
# Radius vs time
# ------------------
r = np.sqrt(x**2 + y**2 + z**2)
plt.figure(figsize=(8,4))
plt.plot(t*1e6, r*1e3)
plt.xlabel("time (µs)")
plt.ylabel("radius |r| (mm)")
plt.title("Ion radial distance vs time")
plt.tight_layout()
plt.show()
