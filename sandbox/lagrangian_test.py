#!/usr/bin/env python3
"""
harmonic_action.py

Compute and visualize the kinetic energy K(t), potential energy V(t),
Lagrangian L(t) = K - V, and the action S = ∫ L dt for a simple harmonic oscillator.

Defaults: m=k=A=1, phi=0, t in [0, 10], action interval [2.0, 7.5].
Usage examples:
    python harmonic_action.py
    python harmonic_action.py --m 2 --k 8 --A 0.3 --phi 0.5 --t_end 20 --t1 3 --t2 12 --n 5000
    python harmonic_action.py --no-3d

The action S is printed both numerically (trapezoid rule) and analytically.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def compute_energies(m, k, A, phi, t):
    omega = np.sqrt(k/m)
    x = A * np.cos(omega * t + phi)
    v = -A * omega * np.sin(omega * t + phi)
    K = 0.5 * m * v**2
    V = 0.5 * k * x**2
    L = K - V
    return omega, x, v, K, V, L

def analytic_action(m, A, omega, phi, t1, t2):
    # For x(t)=A cos(ω t + φ): L = -(1/2) m A^2 ω^2 cos(2ω t + 2φ)
    # S = ∫ L dt = -(1/4) m A^2 ω [sin(2ω t + 2φ)]_{t1}^{t2}
    return -(1/4) * m * A**2 * omega * (np.sin(2*omega*t2 + 2*phi) - np.sin(2*omega*t1 + 2*phi))

def main():
    p = argparse.ArgumentParser(description="Harmonic oscillator: K, V, L, and action S.")
    p.add_argument("--m", type=float, default=1.0, help="Mass m")
    p.add_argument("--k", type=float, default=1.0, help="Spring constant k")
    p.add_argument("--A", type=float, default=1.0, help="Amplitude A")
    p.add_argument("--phi", type=float, default=0.0, help="Phase φ (radians)")
    p.add_argument("--t_end", type=float, default=10.0, help="End time of simulation (start is 0)")
    p.add_argument("--t1", type=float, default=2.0, help="Start time for action integral")
    p.add_argument("--t2", type=float, default=7.5, help="End time for action integral")
    p.add_argument("--n", type=int, default=3000, help="Number of time samples")
    p.add_argument("--no-3d", dest="show3d", action="store_false", help="Disable the 3D (t,K,V) curve")
    p.set_defaults(show3d=True)
    args = p.parse_args()

    t = np.linspace(0.0, args.t_end, args.n)
    omega, x, v, K, V, L = compute_energies(args.m, args.k, args.A, args.phi, t)

    # Action interval handling
    t1 = float(args.t1)
    t2 = float(args.t2)
    if t2 <= t1:
        raise ValueError("Require t2 > t1 for action interval.")
    mask = (t >= t1) & (t <= t2)

    # Numerical action
    S_num = np.trapz(L[mask], t[mask])
    # Analytic action
    S_ana = analytic_action(args.m, args.A, omega, args.phi, t1, t2)

    print(f"Parameters: m={args.m}, k={args.k}, A={args.A}, phi={args.phi}, omega={omega:.6f}")
    print(f"Action interval: [{t1}, {t2}]")
    print(f"Numerical S = {S_num:.6f}")
    print(f"Analytic  S = {S_ana:.6f}")

    # 1) K(t)
    plt.figure()
    plt.plot(t, K)
    plt.xlabel("t (s)")
    plt.ylabel("Kinetic energy K(t)")
    plt.title("K(t) for simple harmonic oscillator")

    # 2) V(t)
    plt.figure()
    plt.plot(t, V)
    plt.xlabel("t (s)")
    plt.ylabel("Potential energy V(t)")
    plt.title("V(t) for simple harmonic oscillator")

    # 3) L(t) with shaded action
    plt.figure()
    plt.plot(t, L)
    plt.xlabel("t (s)")
    plt.ylabel("Lagrangian L(t)")
    plt.title("L(t) with action S shaded")
    plt.fill_between(t[mask], L[mask], step="pre", alpha=0.3)
    plt.axvline(t1, linestyle="--")
    plt.axvline(t2, linestyle="--")

    # 4) Optional 3D curve (t, K, V)
    if args.show3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(t, K, V)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("K(t)")
        ax.set_zlabel("V(t)")
        ax.set_title("3D path: (t, K(t), V(t))")

    plt.show()

if __name__ == "__main__":
    main()
