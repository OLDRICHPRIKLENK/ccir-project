#!/usr/bin/env python3
"""
sho_goal_aware.py

Goal-aware local stepping for a simple harmonic oscillator (SHO).

At each step, we solve for the sinusoid that hits the current point (t_k, x_k)
and the final target (t_f, x_f), then evaluate that sinusoid at t_{k+1}.
This reproduces the exact boundary-value solution for the SHO (up to floating error).

Usage examples:
    python sho_goal_aware.py
    python sho_goal_aware.py --m 1 --k 1 --t0 0 --tf 7 --x0 0.8 --xf -0.2 --steps 300
    python sho_goal_aware.py --no-plot   # skip plotting; prints summary only
"""

import argparse
import numpy as np
import sys

def function(omega, t0, tf, x0, xf, n_steps):
    t = np.linspace(t0, tf, n_steps+1)
    x = np.zeros_like(t)
    x[0] = x0
    for k in range(n_steps):
        tk = t[k]
        tk1 = t[k+1]
        M = np.array([[np.cos(omega*tk), np.sin(omega*tk)],
                      [np.cos(omega*tf), np.sin(omega*tf)]])
        b = np.array([x[k], xf])
        det = np.linalg.det(M)
        if abs(det) < 1e-12:
            raise RuntimeError(

            )
        c1, c2 = np.linalg.solve(M, b)
        x[k+1] = c1*np.cos(omega*tk1) + c2*np.sin(omega*tk1)
    return t, x

def analytic_two_point(omega, t0, tf, x0, xf, tgrid):
    M = np.array([[np.cos(omega*t0), np.sin(omega*t0)],
                  [np.cos(omega*tf), np.sin(omega*tf)]])
    b = np.array([x0, xf])
    det = np.linalg.det(M)
    if abs(det) < 1e-12:
        raise RuntimeError(
            "Endpoints ill-conditioned: sin(ω(tf - t0))≈0. "
            "Choose tf so ω(tf - t0) is not an integer multiple of π."
        )
    c1, c2 = np.linalg.solve(M, b)
    x = c1*np.cos(omega*tgrid) + c2*np.sin(omega*tgrid)
    return x

def main():
    ap = argparse.ArgumentParser(description="Goal-aware local stepping for the simple harmonic oscillator.")
    ap.add_argument("--m", type=float, default=1.0, help="mass m")
    ap.add_argument("--k", type=float, default=1.0, help="spring constant k")
    ap.add_argument("--t0", type=float, default=0.0, help="start time")
    ap.add_argument("--tf", type=float, default=7.0, help="final time")
    ap.add_argument("--x0", type=float, default=0.8, help="x(t0)")
    ap.add_argument("--xf", type=float, default=-0.2, help="x(tf)")
    ap.add_argument("--steps", type=int, default=300, help="number of steps")
    ap.add_argument("--no-plot", action="store_true", help="skip plotting")
    args = ap.parse_args()

    omega = np.sqrt(args.k/args.m)
    t = np.linspace(args.t0, args.tf, args.steps+1)

    # Compute paths
    t_goal, x_goal = goal_aware_step_path(omega, args.t0, args.tf, args.x0, args.xf, args.steps)
    x_true = analytic_two_point(omega, args.t0, args.tf, args.x0, args.xf, t)

    # Report
    max_dev = float(np.max(np.abs(x_goal - x_true)))
    print("=== SHO goal-aware stepping ===")
    print(f"m={args.m}, k={args.k}, omega={omega:.6f}")
    print(f"t0={args.t0}, tf={args.tf}, steps={args.steps}")
    print(f"x0={args.x0}, xf={args.xf}")
    print(f"Max |goal-aware - exact| = {max_dev:.3e}")

    if args.no_plot:
        return 0

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, x_true, label="Exact two-point path")
        plt.plot(t_goal, x_goal, linestyle="--", label="Goal-aware local stepping")
        plt.xlabel("t (s)")
        plt.ylabel("x(t)")
        plt.title("Goal-aware local stepping vs. exact solution (SHO)")
        plt.legend()
        plt.show()
    except Exception as e:
        print("Plotting failed (run with --no-plot to suppress). Error:", e, file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
