# Ion trap simulator (Ca+) — hyperbolic OR linear Paul trap + optional Doppler cooling
# Paste into a notebook. Requires: numpy, scipy, matplotlib, pandas.
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# =========================
# USER PARAMETERS
# =========================
# Ion (Ca+)
m = 6.64e-26          # kg  (~40 amu)
e = 1.602176634e-19   # C

# ---- Choose trap mode ----
trap_mode = "linear"   # "hyperbolic" or "linear"

# Common RF drive
Omega = 2.0e7          # rad/s (RF angular frequency)
Vrf   = 200.0          # V (RF amplitude)

# Hyperbolic Paul trap geometry & DC
r0_h      = 3.0e-3     # m (characteristic radius)
Udc_h     = 0.0        # V (DC bias, often 0)

# Linear Paul trap geometry & DC (axial confinement)
r0_l      = 1.5e-3     # m (radial scale)
z0_l      = 3.0e-3     # m (axial scale to endcaps)
kappa_l   = 0.30       # geometry factor (0.2–0.5 typical)
Uec_l     = 5.0        # V (endcap DC for axial harmonic well)

# Optional viscous damping (buffer gas, “generic” cooling)
gamma = 0.0            # s^-1 (0 = off)

# ---- Laser cooling (Doppler) ----
cooling_mode = "doppler"   # "none" or "doppler"
# Cooling beams per axis (bools): counter-propagating ±k along chosen axes
cool_x, cool_y, cool_z = True, True, True

# Ca+ 397 nm (S1/2 -> P1/2) cycling transition (simplified 1D per axis)
hbar   = 1.054_571_817e-34
lambda0= 397e-9                         # m
k0     = 2*np.pi/lambda0                # 1/m
Gamma  = 2*np.pi*21.6e6                 # s^-1 (natural linewidth ~21.6 MHz)
s0     = 0.5                            # saturation parameter of each beam
Delta  = -Gamma/2                       # detuning (red, ~ -Gamma/2 good for Doppler)

# Initial conditions (m, m/s)
x0, y0, z0    = 0.3e-3, 0.0e-3, 0.0e-3
vx0, vy0, vz0 = 0.0,    0.0,    0.0

# Integration controls
n_rf_periods      = 400
steps_per_period  = 500
save_csv          = True
csv_path          = "ion_trap_timeseries.csv"
# =========================

# =========================
# Potentials and gradients
# =========================
def phi_hyperbolic(x,y,z,t):
    # Φ = ((U + Vrf cosΩt) / (2 r0^2)) (x^2 + y^2 - 2 z^2)
    coeff = (Udc_h + Vrf*np.cos(Omega*t)) / (2*r0_h**2)
    return coeff*(x*x + y*y - 2*z*z)

def grad_phi_hyperbolic(x,y,z,t):
    c = (Udc_h + Vrf*np.cos(Omega*t)) / (r0_h**2)
    return c*x, c*y, -2*c*z

def phi_linear(x,y,z,t):
    # RF: Φ_rf ≈ (Vrf cosΩt / (2 r0^2)) (x^2 - y^2)
    # DC axial: Φ_dc ≈ (κ Uec / z0^2) z^2
    return (Vrf*np.cos(Omega*t)/(2*r0_l**2))*(x*x - y*y) + (kappa_l*Uec_l/z0_l**2)*z*z

def grad_phi_linear(x,y,z,t):
    crf = Vrf*np.cos(Omega*t)/(r0_l**2)
    cdc = (kappa_l*Uec_l)/(z0_l**2)
    dphidx = crf*x
    dphidy = -crf*y
    dphidz = 2*cdc*z
    return dphidx, dphidy, dphidz

if trap_mode == "hyperbolic":
    phi, grad_phi = phi_hyperbolic, grad_phi_hyperbolic
elif trap_mode == "linear":
    phi, grad_phi = phi_linear, grad_phi_linear
else:
    raise ValueError("trap_mode must be 'hyperbolic' or 'linear'.")

# =========================
# Doppler cooling force (per axis, counter-propagating beams)
# =========================
def doppler_force_component(v):
    """
    1D radiation-pressure force along one axis from ±k beams:
    F = ħk * Γ * (s/2) [ 1/(1+s + (2(Δ - kv)/Γ)^2) - 1/(1+s + (2(Δ + kv)/Γ)^2) ]
    """
    if s0 <= 0:
        return 0.0
    # Scattering rates from +k and -k beams
    denom_plus  = 1 + s0 + (2*(Delta - k0*v)/Gamma)**2
    denom_minus = 1 + s0 + (2*(Delta + k0*v)/Gamma)**2
    R_plus  = (Gamma * s0 / 2) / denom_plus
    R_minus = (Gamma * s0 / 2) / denom_minus
    return hbar*k0*(R_plus - R_minus)

def cooling_force(vx,vy,vz):
    if cooling_mode != "doppler":
        return 0.0, 0.0, 0.0
    Fx = doppler_force_component(vx) if cool_x else 0.0
    Fy = doppler_force_component(vy) if cool_y else 0.0
    Fz = doppler_force_component(vz) if cool_z else 0.0
    return Fx, Fy, Fz

# =========================
# ODEs from Euler–Lagrange: m r̈ + e ∇Φ + γ ṙ = F_laser
# =========================
def rhs(t, Y):
    x,y,z, vx,vy,vz = Y
    gx,gy,gz = grad_phi(x,y,z,t)           # ∇Φ
    Fx_l, Fy_l, Fz_l = cooling_force(vx,vy,vz)
    ax = (-e*gx - gamma*vx + Fx_l)/m
    ay = (-e*gy - gamma*vy + Fy_l)/m
    az = (-e*gz - gamma*vz + Fz_l)/m
    return [vx, vy, vz, ax, ay, az]

# =========================
# Integrate
# =========================
T_rf   = 2*pi/Omega
t_end  = n_rf_periods*T_rf
t_eval = np.linspace(0.0, t_end, int(n_rf_periods*steps_per_period)+1)
Y0     = [x0,y0,z0,vx0,vy0,vz0]

# Print handy frequencies (small-q secular estimates for intuition only)
if trap_mode == "hyperbolic":
    a_xy = (4*e*Udc_h)/(m*r0_h**2*Omega**2); q_xy = (2*e*Vrf)/(m*r0_h**2*Omega**2)
    a_z  = -2*a_xy; q_z = -2*q_xy
    w_sec_xy = abs(q_xy)*Omega/(2*np.sqrt(2)); w_sec_z = abs(q_z)*Omega/(2*np.sqrt(2))
    print(f"[Hyperbolic] a_xy={a_xy:.3e}, q_xy={q_xy:.3e}; a_z={a_z:.3e}, q_z={q_z:.3e}")
    print(f"  Secular freq (approx): wr≈{w_sec_xy:.2e} rad/s, wz≈{w_sec_z:.2e} rad/s")
else:
    # Radial secular (approx) in linear trap from q≈2eVrf/(m r0^2 Ω^2)
    q_r = (2*e*Vrf)/(m*r0_l**2*Omega**2)
    w_r = abs(q_r)*Omega/(2*np.sqrt(2))
    # Axial harmonic from DC
    w_z = np.sqrt( 2*e*kappa_l*Uec_l/(m*z0_l**2) )
    print(f"[Linear] q_r≈{q_r:.3e}  -> wr≈{w_r:.2e} rad/s;  wz≈{w_z:.2e} rad/s (axial)")

sol = solve_ivp(rhs, (0.0, t_end), Y0, method="RK45", t_eval=t_eval, rtol=1e-7, atol=1e-10)
if not sol.success:
    print("Integrator warning:", sol.message)

t = sol.t
x,y,z,vx,vy,vz = sol.y
r = np.sqrt(x*x + y*y + z*z)

# Energies
K = 0.5*m*(vx*vx + vy*vy + vz*vz)
V = e*phi(x,y,z,t)
E = K + V

# Heuristic trapping classifier
def envelope_max(arr, fraction=0.25):
    n = len(arr); k = max(1, int(fraction*n))
    return np.max(np.abs(arr[:k])), np.max(np.abs(arr[-k:]))

rmax_start, rmax_end = envelope_max(r, 0.25)
growth = (rmax_end+1e-30)/(rmax_start+1e-30)
trapped = np.isfinite(rmax_end) and growth < 1.5

print("\nHeuristic trapping assessment:")
print(f"  Max |r| (start): {rmax_start:.3e} m")
print(f"  Max |r| (end)  : {rmax_end:.3e} m")
print(f"  Growth factor   : {growth:.3f}")
print(f"  => { 'TRAPPED (bounded)' if trapped else 'UNSTABLE / ESCAPING' }")
if cooling_mode == "doppler" or gamma > 0:
    print("  (Cooling on: expect secular amplitude/energy to decrease.)")

# =========================
# Plots
# =========================
plt.figure(figsize=(8,4))
plt.plot(t*1e6, r*1e3)
plt.xlabel("time (µs)"); plt.ylabel("radius |r| (mm)")
plt.title(f"Ion radius vs time ({trap_mode} trap, cooling={cooling_mode})")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(t*1e6, x*1e3, label='x'); plt.plot(t*1e6, y*1e3, label='y'); plt.plot(t*1e6, z*1e3, label='z')
plt.xlabel("time (µs)"); plt.ylabel("position (mm)")
plt.title("Position components vs time"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(t*1e6, E)
plt.xlabel("time (µs)"); plt.ylabel("Total energy (J)")
plt.title("Total energy vs time"); plt.tight_layout(); plt.show()

plt.figure(figsize=(5,5))
plt.plot(x*1e3, vx); plt.xlabel("x (mm)"); plt.ylabel("vx (m/s)")
plt.title("Phase portrait: x vs vx"); plt.tight_layout(); plt.show()

if save_csv:
    df = pd.DataFrame({
        "t_s": t,
        "x_m": x, "y_m": y, "z_m": z,
        "vx_m_s": vx, "vy_m_s": vy, "vz_m_s": vz,
        "r_m": r, "K_J": K, "V_J": V, "E_J": E
    })
    df.to_csv(csv_path, index=False)
    print(f"\nSaved timeseries to: {csv_path}")

# =========================
# Notes
# =========================
# • Switch trap by setting trap_mode = "hyperbolic" or "linear".
# • For linear trap, axial confinement from Uec_l; radial from RF.
# • Doppler cooling is a simple 1D-per-axis model; for low v it behaves ~ -α v.
#   Tune (Delta, s0) and choose which axes have beams with (cool_x, cool_y, cool_z).
# • If you also set gamma > 0, that adds extra viscous damping (e.g., buffer gas).
# • Increase steps_per_period if RF micromotion looks under-resolved.
