import numpy as np
from scipy.special import genlaguerre, sph_harm
from scipy.constants import hbar, physical_constants, epsilon_0, e, pi

# -------------------------
# CONFIGURABLE PARAMETERS
# -------------------------
n = 1         # Principal quantum number (n >= 1)
l = 1         # Orbital quantum number (0 <= l < n)
m = 1         # Magnetic quantum number (-l <= m <= l)
Z = 1         # Atomic number (Z = 1 for hydrogen)
r_max = 20e-10 # Maximum radius in meters
num_points = 500  # Number of points for radial array

# -------------------------
# CONSTANTS
# -------------------------
a0 = physical_constants["Bohr radius"][0]           # Bohr radius (m)
E_h = physical_constants["Hartree energy"][0]       # Hartree energy (J)
eV = physical_constants["electron volt"][0]         # 1 eV in J

# -------------------------
# ENERGY LEVEL
# -------------------------
def schrodinger_energy(n, Z=1):
    return - (Z**2) * E_h / n**2  # Energy in J

energy_joules = schrodinger_energy(n, Z)
energy_ev = energy_joules / eV

print(f"Schrödinger Energy Level for n={n}, l={l}:")
print(f"  Energy = {energy_joules:.5e} J")
print(f"  Energy = {energy_ev:.5f} eV")

