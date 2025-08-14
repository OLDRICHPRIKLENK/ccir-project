import numpy as np
import matplotlib.pyplot as plt

g = 9.81

def endpoint_equation(K, yA, yB, yk):
    """Horizontal distance from current point yk to yB for given K (start from rest)."""
    hB = yA - yB
    hk = yA - yk
    term1 = K**2 * (np.arcsin(np.sqrt(hB)/K) - np.arcsin(np.sqrt(hk)/K))
    term2 = np.sqrt(hB * (K**2 - hB)) - np.sqrt(hk * (K**2 - hk))
    return term1 - term2

def solve_for_K(yA, yB, yk, x_rem):
    """Solve endpoint equation for K given remaining x distance."""
    hB = yA - yB
    hk = yA - yk
    # Ensure we start above the singularity
    K_low = max(np.sqrt(hB), np.sqrt(hk)) * (1 + 1e-12)
    K_high = K_low * 2.0
    # Expand high bound until it overshoots x_rem
    while endpoint_equation(K_high, yA, yB, yk) < x_rem:
        K_high *= 2.0
    # Bisection
    for _ in range(100):
        K_mid = 0.5 * (K_low + K_high)
        f_mid = endpoint_equation(K_mid, yA, yB, yk)
        if f_mid < x_rem:
            K_low = K_mid
        else:
            K_high = K_mid
        if abs(f_mid - x_rem) < 1e-12 * max(1.0, x_rem):
            break
    return 0.5 * (K_low + K_high)

def goal_aware_stepwise(xA, yA, xB, yB, N=500):
    """Simulate the process: at each step, pick K to connect current point to goal, step Δx."""
    xs = np.zeros(N+1)
    ys = np.zeros(N+1)
    xs[0], ys[0] = xA, yA - 1e-6  # seed small drop to avoid infinite slope
    dx = (xB - xA) / N
    for k in range(N):
        xk, yk = xs[k], ys[k]
        x_rem = xB - xk
        if x_rem <= 0:
            xs[k+1:] = xB
            ys[k+1:] = yB
            break
        Kk = solve_for_K(yA, yB, yk, x_rem)
        hk = yA - yk
        slope = -np.sqrt(Kk**2 / hk - 1.0)
        xs[k+1] = xk + dx
        ys[k+1] = yk + slope * dx
    return xs, ys

# Example start & end points
xA, yA = 0.0, 0.0
xB, yB = 2.0, -2.0

# Run the stepwise solver
xs, ys = goal_aware_stepwise(xA, yA, xB, yB, N=800)

# Analytical cycloid for comparison
def analytic_brachistochrone_params(xA, yA, xB, yB):
    drop = yA - yB
    theta = np.pi / 2
    for _ in range(80):
        R = drop / (1 - np.cos(theta))
        f = R * (theta - np.sin(theta)) - (xB - xA)
        df = (drop * np.sin(theta) / (1 - np.cos(theta))**2) * (theta - np.sin(theta)) + R * (1 - np.cos(theta))
        theta -= f / df
    theta_f = theta
    R = drop / (1 - np.cos(theta_f))
    return R, theta_f

R, theta_f = analytic_brachistochrone_params(xA, yA, xB, yB)
th = np.linspace(0, theta_f, 1000)
x_an = xA + R * (th - np.sin(th))
y_an = yA - R * (1 - np.cos(th))

# Plot
plt.figure(figsize=(7,5))
plt.plot(x_an, y_an, label='Analytical cycloid', lw=2)
plt.plot(xs, ys, '--', label='Goal-aware stepwise', lw=1)
plt.scatter([xA, xB], [yA, yB], c='red', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Goal-Aware Stepwise vs Analytical Brachistochrone')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
