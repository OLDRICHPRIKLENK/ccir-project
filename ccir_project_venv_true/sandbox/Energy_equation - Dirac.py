import numpy as np

# ---------------------------
# CONFIGURABLE PARAMETERS
# ---------------------------
Z = 1                 # Atomic number (e.g., Z = 1 for H, 20 for Ca20+)
n = 1                  # Principal quantum number (n = 1, 2, 3, ...)
l = 1                 # Orbital angular momentum quantum number (l = 0, 1, ..., n-1)
j = 1               # Total angular momentum (j = l ± 1/2)
m_e = 9.10938356e-31   # Electron mass (kg)
c = 299792458          # Speed of light (m/s)
hbar = 1.054571817e-34 # Reduced Planck's constant (J·s)
alpha = 1/137.035999   # Fine-structure constant
eV = 1.602176634e-19   # Electron volt (J)

# ---------------------------
# DERIVED QUANTITIES
# ---------------------------
# Dirac quantum number κ
kappa = -(int(j + 0.5)) if j == l + 0.5 else int(j + 0.5)

# Check for physical validity
Z_alpha = Z * alpha
if abs(Z_alpha) >= abs(kappa):
    raise ValueError("Invalid parameters: sqrt of negative number in Dirac equation")

# ---------------------------
# DIRAC ENERGY CALCULATION
# ---------------------------
def dirac_energy(Z, n, kappa, m_e, c, alpha):
    term = Z * alpha
    gamma = np.sqrt(kappa**2 - term**2)
    denom = n - abs(kappa) + gamma
    energy = m_e * c**2 / np.sqrt(1 + (term / denom)**2)
    return energy

# Calculate energy in joules and eV
E_joules = dirac_energy(Z, n, kappa, m_e, c, alpha)
E_eV = E_joules / eV

# ---------------------------
# OUTPUT
# ---------------------------
print(f"Dirac Energy Level for Z={Z}, n={n}, l={l}, j={j}:")
print(f"  Energy = {E_joules:.5e} J")
print(f"  Energy = {E_eV:.5f} eV")
