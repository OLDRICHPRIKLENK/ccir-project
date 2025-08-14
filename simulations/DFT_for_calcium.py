"""
Minimal radial DFT (spherical atoms) with LDA XC — SciPy >= 1.11 compatible.
Uses scipy.integrate.trapezoid and cumulative_trapezoid exclusively.

Dependencies:
    numpy, scipy (linalg, integrate), matplotlib
"""

import numpy as np
from numpy import pi
from scipy import linalg
from scipy.integrate import trapezoid, cumulative_trapezoid
import matplotlib.pyplot as plt


def dft_atom(Z, N_electrons,
             r_max=20.0, N_grid=1000,
             max_iter=1000, tolerance=1e-8, mixing=0.3,
             verbose=True, make_plots=True):
    """
    Self-consistent DFT for a spherical atom on a radial grid.

    Args:
        Z: nuclear charge
        N_electrons: total number of electrons
        r_max: radial box size (a.u.)
        N_grid: number of radial grid points
        max_iter: SCF iteration cap
        tolerance: energy change threshold for convergence
        mixing: simple linear density mixing
        verbose: print SCF table if True
        make_plots: generate figure if True
    """
    # --- Radial grid (avoid r=0 to prevent 1/r blowups) ---
    dr = r_max / N_grid
    r = np.linspace(dr, r_max, N_grid)

    # --- Initial density guess: hydrogenic-like exponential; normalize to N ---
    rho = (N_electrons *
           np.exp(-2 * Z * r / N_electrons) *
           (2 * Z / N_electrons) ** 3 / (8 * pi))
    rho = rho / trapezoid(4 * pi * r**2 * rho, x=r) * N_electrons

    if verbose:
        print("=== DFT Implementation Test ===")
        print(f"Z={Z}, N_electrons={N_electrons}")
        print("Iter\tTotal Energy\tKinetic\t\tNuclear\t\tHartree\t\tXC\t\tConvergence")

    E_total_old = 0.0

    for it in range(1, max_iter + 1):
        # --- Build effective potential V_eff(r) = V_nuclear + V_H + V_xc ---
        V_eff = build_effective_potential(r, rho, Z)

        # --- Solve Kohn–Sham radial equations on the grid ---
        psi_orbs, E_orbs, occupations = solve_kohn_sham(r, V_eff, N_electrons)

        # --- Density from occupied orbitals: rho(r) = sum occ * |R(r)|^2 / (4π r^2) ---
        rho_new = np.zeros_like(r)
        for i, occ in enumerate(occupations):
            if occ > 0:
                rho_new += occ * psi_orbs[:, i] ** 2 / (4 * pi * r ** 2)

        # --- Energy components and total energy ---
        E_kin, E_nuc, E_H, E_xc, E_total = calculate_energies(
            r, rho_new, psi_orbs, E_orbs, occupations, Z
        )

        conv = abs(E_total - E_total_old)
        if verbose:
            print(f"{it}\t{E_total: .8f}\t{E_kin: .8f}\t{E_nuc: .8f}\t{E_H: .8f}\t{E_xc: .8f}\t{conv: .2e}")

        if conv < tolerance:
            if verbose:
                print("Converged!")
                if Z == 1:
                    print(f"Exact H atom energy: -0.5 a.u., Error: {E_total + 0.5: .6f} a.u.")
                elif Z == 2:
                    print(f"Experimental He energy: ~-2.903 a.u., Error: {E_total + 2.903: .6f} a.u.")
            break

        # --- Simple linear mixing for stability + renormalize ---
        rho = (1 - mixing) * rho + mixing * rho_new
        rho = rho / trapezoid(4 * pi * r**2 * rho, x=r) * N_electrons
        E_total_old = E_total
    else:
        if verbose:
            print("Warning: Did not converge!")

    if make_plots:
        plot_results(r, rho_new, psi_orbs, occupations, Z, N_electrons, E_total, E_orbs)

    return {
        "r": r,
        "rho": rho_new,
        "psi_orbs": psi_orbs,
        "E_orbs": E_orbs,
        "occupations": occupations,
        "energies": {
            "total": E_total,
            "kinetic": E_kin,
            "nuclear": E_nuc,
            "hartree": E_H,
            "xc": E_xc,
        },
        "converged": (conv < tolerance)
    }


# ===================== POTENTIALS & ENERGIES =====================

def build_effective_potential(r, rho, Z):
    """V_eff = V_nuclear + V_Hartree + V_xc (LDA)."""
    V_nuclear = -Z / r
    V_hartree = calculate_hartree_potential(r, rho)
    V_xc = calculate_xc_potential_lda(rho)
    return V_nuclear + V_hartree + V_xc


def calculate_hartree_potential(r, rho):
    """
    Spherical Poisson via analytic integrals for radial symmetry:
      V(r) = (1/r) ∫_0^r 4π r'^2 ρ(r') dr' + ∫_r^∞ 4π r' ρ(r') dr'

    Computed with cumulative trapezoids (forward and reverse).
    """
    fourpi = 4 * pi

    # Charge enclosed up to r: Q_in(r) = ∫_0^r 4π r'^2 ρ(r') dr'
    Q_in = cumulative_trapezoid(fourpi * r**2 * rho, x=r, initial=0.0)

    # Outer integral: I_out(r) = ∫_r^∞ 4π r' ρ(r') dr'
    integrand_out = fourpi * r * rho
    I_out_rev = cumulative_trapezoid(integrand_out[::-1], x=r[::-1], initial=0.0)
    I_out = I_out_rev[::-1]

    V = Q_in / r + I_out
    return V


def calculate_xc_potential_lda(rho):
    """
    LDA:
      V_x: exact exchange (UEG)
      V_c: Perdew–Zunger parameterization of Ceperley–Alder data
    """
    rho_safe = np.maximum(rho, 1e-30)
    rs = (3.0 / (4 * pi * rho_safe)) ** (1.0 / 3.0)

    # Exchange potential
    V_x = -(3 / pi) ** (1 / 3) * rho_safe ** (1 / 3)

    # Perdew–Zunger correlation (two-branch)
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334

    V_c = np.zeros_like(rho_safe)
    mask_high = rs < 1
    if np.any(mask_high):
        lnrs = np.log(rs[mask_high])
        V_c[mask_high] = A * lnrs + B + C * rs[mask_high] * lnrs + D * rs[mask_high]
    mask_low = ~mask_high
    if np.any(mask_low):
        sqrt_rs = np.sqrt(rs[mask_low])
        V_c[mask_low] = gamma / (1 + beta1 * sqrt_rs + beta2 * rs[mask_low])

    V_xc = V_x + V_c
    V_xc[~np.isfinite(V_xc)] = 0.0
    V_xc[rho < 1e-12] = 0.0
    return V_xc


def calculate_xc_energy_lda(r, rho):
    """
    LDA exchange-correlation total energy:
      E_xc = ∫ 4π r^2 ρ(r) ε_xc[ρ(r)] dr
    """
    rho_safe = np.maximum(rho, 1e-30)
    rs = (3.0 / (4 * pi * rho_safe)) ** (1.0 / 3.0)

    # Exchange energy density (per volume)
    epsilon_x = -(3.0 / 4.0) * (3 / pi) ** (1 / 3) * rho_safe ** (4.0 / 3.0)

    # Perdew–Zunger correlation energy density (per volume)
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
    epsilon_c = np.zeros_like(rho_safe)

    mask_high = rs < 1
    if np.any(mask_high):
        lnrs = np.log(rs[mask_high])
        epsilon_c[mask_high] = A * lnrs + B + C * rs[mask_high] * lnrs + D * rs[mask_high]
    mask_low = ~mask_high
    if np.any(mask_low):
        sqrt_rs = np.sqrt(rs[mask_low])
        epsilon_c[mask_low] = gamma / (1 + beta1 * sqrt_rs + beta2 * rs[mask_low])

    epsilon_xc = epsilon_x + epsilon_c
    epsilon_xc[rho < 1e-12] = 0.0

    fourpi = 4 * pi
    E_xc = trapezoid(fourpi * r**2 * rho_safe * epsilon_xc, x=r)
    return E_xc


def calculate_energies(r, rho, psi_orbs, E_orbs, occupations, Z):
    """Compute all energy components from the (updated) density and orbitals."""
    dr = r[1] - r[0]
    fourpi = 4 * pi

    # Kinetic energy via ∫ (1/2) |∇ψ|^2 r^2 dr for radial functions
    E_kinetic = 0.0
    for i, occ in enumerate(occupations):
        if occ > 0:
            dpsi_dr = np.gradient(psi_orbs[:, i], dr, edge_order=2)
            E_kinetic += occ * 0.5 * trapezoid(dpsi_dr**2 * r**2, x=r)

    # Nuclear attraction: ∫ ρ(r) (-Z/r) d^3r = -Z ∫ 4π r^2 ρ(r) / r dr
    E_nuclear = -Z * trapezoid(fourpi * r**2 * rho / r, x=r)

    # Hartree energy: (1/2) ∫ ρ(r) V_H(r) d^3r
    V_H = calculate_hartree_potential(r, rho)
    E_hartree = 0.5 * trapezoid(fourpi * r**2 * rho * V_H, x=r)

    # XC energy
    E_xc = calculate_xc_energy_lda(r, rho)

    E_total = E_kinetic + E_nuclear + E_hartree + E_xc
    return E_kinetic, E_nuclear, E_hartree, E_xc, E_total


# ===================== KS SOLVER =====================

def solve_kohn_sham(r, V_eff, N_electrons):
    """
    Build the finite-difference radial Hamiltonian and solve for lowest states.
    We do a simple dense symmetric eigensolve (SciPy linalg.eigh) for clarity.
    """
    N = len(r)
    dr = r[1] - r[0]

    # Second-derivative operator with Dirichlet-like boundaries
    T = np.zeros((N, N))
    idx = np.arange(1, N - 1)
    T[idx, idx - 1] = 1.0 / (dr ** 2)
    T[idx, idx]     = -2.0 / (dr ** 2)
    T[idx, idx + 1] = 1.0 / (dr ** 2)
    T[0, 0]  = -2.0 / (dr ** 2); T[0, 1]   =  1.0 / (dr ** 2)
    T[-1, -2]=  1.0 / (dr ** 2); T[-1, -1] = -2.0 / (dr ** 2)
    T = -0.5 * T

    H = T + np.diag(V_eff)

    # Solve symmetric eigenproblem
    E_all, psi_all = linalg.eigh(H)
    idx_sort = np.argsort(E_all)
    E_all = np.real(E_all[idx_sort])
    psi_all = np.real(psi_all[:, idx_sort])

    # Occupations: fill in order, 2 electrons per spatial orbital (spin paired)
    n_orbitals = int(np.ceil(N_electrons / 2))
    if N_electrons > 10:
        n_orbitals = min(n_orbitals, 15)  # safety cap like MATLAB

    occupations = np.zeros(n_orbitals)
    remaining = N_electrons
    for i in range(n_orbitals):
        if remaining >= 2:
            occupations[i] = 2.0
            remaining -= 2
        elif remaining > 0:
            occupations[i] = remaining
            remaining = 0
        else:
            occupations[i] = 0.0

    psi_orbs = psi_all[:, :n_orbitals]
    E_orbs = E_all[:n_orbitals]

    # Normalize radial wavefunctions: ∫ |R(r)|^2 r^2 dr = 1
    for i in range(n_orbitals):
        s = trapezoid(psi_orbs[:, i] ** 2 * r ** 2, x=r)
        if s > 1e-10:
            psi_orbs[:, i] /= np.sqrt(s)

    return psi_orbs, E_orbs, occupations


# ===================== PLOTTING =====================

def plot_results(r, rho, psi_orbs, occupations, Z, N_electrons, E_total, E_orbs):
    fig = plt.figure(figsize=(12, 8))

    # Density (log scale)
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(r, rho, lw=2)
    ax1.set_xlabel('r (a.u.)'); ax1.set_ylabel(r'$\rho(r)$')
    ax1.set_title('Electron Density')
    ax1.grid(True); ax1.set_xlim(0, 10)

    # Radial density distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(r, 4 * pi * r ** 2 * rho, lw=2)
    ax2.set_xlabel('r (a.u.)'); ax2.set_ylabel(r'$4\pi r^2 \rho(r)$')
    ax2.set_title('Radial Density Distribution')
    ax2.grid(True); ax2.set_xlim(0, 10)

    # Orbitals
    ax3 = plt.subplot(2, 3, 3)
    nplot = min(4, psi_orbs.shape[1])
    labels = []
    for i in range(nplot):
        if occupations[i] > 0:
            ax3.plot(r, psi_orbs[:, i], lw=2)
            labels.append(f"Orbital {i+1} (occ={occupations[i]:.1f})")
    ax3.set_xlabel('r (a.u.)'); ax3.set_ylabel('R(r)')
    ax3.set_title('Radial Orbitals')
    if labels:
        ax3.legend(labels, loc='best')
    ax3.grid(True); ax3.set_xlim(0, 10)

    # Potentials
    ax4 = plt.subplot(2, 3, 4)
    V_nuclear = -Z / r
    V_hartree = calculate_hartree_potential(r, rho)
    V_xc = calculate_xc_potential_lda(rho)
    V_eff = V_nuclear + V_hartree + V_xc
    ax4.plot(r, V_nuclear, ls='--', lw=1.5, label='Nuclear')
    ax4.plot(r, V_hartree, ls=':', lw=1.5, label='Hartree')
    ax4.plot(r, V_xc, ls=':', lw=1.5, label='XC')
    ax4.plot(r, V_eff, lw=2, label='Total')
    ax4.set_xlabel('r (a.u.)'); ax4.set_ylabel('V(r)')
    ax4.set_title('Effective Potential Components')
    ax4.legend(loc='best'); ax4.grid(True)
    ax4.set_xlim(0, 8); ax4.set_ylim(-5, 2)

    # Text panel (energies / info)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    lines = [
        f"Z = {Z}, N = {N_electrons}",
        f"E_total = {E_total:.8f} a.u.",
        *[f"ε_{i+1} = {E_orbs[i]:.6f} a.u., occ = {occupations[i]:.1f}"
          for i in range(min(len(E_orbs), 6))]
    ]
    ax5.text(0.0, 0.95, "\n".join(lines), va='top', ha='left', fontsize=11)

    plt.tight_layout()
    plt.show()


# ===================== Quick test (Hydrogen) =====================

if __name__ == "__main__":
    # Example: Hydrogen atom test (Z=1, N=1)
    dft_atom(Z=2, N_electrons=2)
