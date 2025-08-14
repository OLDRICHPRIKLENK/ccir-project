import numpy as np
import matplotlib.pyplot as plt

# Constants
m, c = 1.0, 1.0
T, N = 20.0, 800
t = np.linspace(0, T, N)

# Target transverse speed profile (choose what you like)
v0, a = 0.2, 0.04             # v_perp(t) = v0 + a t  (linear growth)
v = v0 + a*t                  # SAME for both scenarios

# --- Scenario A: grow radius, keep omega fixed ---
omega_A = 2*np.pi/8           # fixed
R_A = v / omega_A             # choose R(t) so that R*omega = v
theta_A = np.cumsum(omega_A*np.ones_like(t)) * (t[1]-t[0])
xA = R_A*np.cos(theta_A)
yA = R_A*np.sin(theta_A)

# --- Scenario B: keep radius fixed, grow omega ---
R_B = 1.0                     # fixed
omega_B = v / R_B             # choose omega(t) so that R*omega = v
theta_B = np.cumsum(omega_B) * (t[1]-t[0])
xB = R_B*np.cos(theta_B)
yB = R_B*np.sin(theta_B)

# Energies (identical because v is identical)
gamma = 1.0/np.sqrt(1.0 - (v/c)**2)
E = gamma*m*c**2

# --- Plots ---
fig = plt.figure(figsize=(11,6))

# Left: two helices (x,y,t)
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot(xA, yA, t, label="A: R↑, ω const")
ax.plot(xB, yB, t, label="B: R const, ω↑")
ax.set_xlabel("Real space x")
ax.set_ylabel("Imaginary y")
ax.set_zlabel("Time t")
ax.set_title("Two constructions with the SAME v⊥(t)")
ax.legend(loc="upper left")
ax.view_init(elev=22, azim=35)

# Right: energy vs time (curves overlap)
ax2 = fig.add_subplot(1,2,2)
ax2.plot(t, E, label="E(t) for A & B (overlap)")
ax2.set_xlabel("Time t")
ax2.set_ylabel("Total energy E(t)")
ax2.set_title("Energy depends only on v⊥(t), not on (R, ω) split")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
