import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, sph_harm, factorial
from scipy.constants import hbar, epsilon_0, e, pi, physical_constants

# -------------------------
# Constants
# -------------------------
m_e = physical_constants["electron mass"][0]
a0 = 4 * pi * epsilon_0 * hbar**2 / (m_e * e**2)

# -------------------------
# Full hydrogen wavefunction |ψ_nlm(x, z)|²
# -------------------------
def hydrogen_density(n, l, m, x, z, phi_fixed=0, Z=1):
    r = np.sqrt(x**2 + z**2)
    theta = np.arccos(np.divide(z, r, out=np.zeros_like(z), where=r != 0))
    rho = 2 * Z * r / (n * a0)
    norm = np.sqrt((2 * Z / (n * a0))**3 * factorial(n - l - 1, exact=True) / (2 * n * factorial(n + l, exact=True)))
    L = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R_radial = norm * np.exp(-rho / 2) * rho**l * L
    Y = sph_harm(m, l, phi_fixed, theta)
    return np.abs(R_radial * Y)**2

# -------------------------
# Grid setup (Cartesian xz-plane)
# -------------------------
grid_size = 500
extent = 20 * a0
x = np.linspace(-extent, extent, grid_size)
z = np.linspace(-extent, extent, grid_size)
X, Z = np.meshgrid(x, z)
PHI_FIXED = 0  # 2D slice through phi = 0

# -------------------------
# Choose quantum numbers
# -------------------------
n, l, m = 3, 2, 1  # Try (2,1,0), (3,2,1), etc.

# -------------------------
# Evaluate probability density
# -------------------------
density = hydrogen_density(n, l, m, X, Z, phi_fixed=PHI_FIXED)
density /= density.max()  # normalize

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6, 6))
plt.imshow(density, extent=[x[0]/1e-10, x[-1]/1e-10, z[0]/1e-10, z[-1]/1e-10],
           cmap='inferno', origin='lower')
plt.xlabel('x (Å)')
plt.ylabel('z (Å)')
plt.title(f'|ψ|² (n={n}, l={l}, m={m}) slice at φ={PHI_FIXED}')
plt.colorbar(label='Probability Density')
plt.axis('equal')
plt.tight_layout()
plt.show()
