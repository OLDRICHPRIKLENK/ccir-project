import sympy as sp

# Define symbols
t, m, g = sp.symbols('t m g', real=True)

# Indexed functions of time
R = sp.Function('R')  # base function
Theta = sp.Function('Theta')

# Indices
i, f = sp.symbols('i f')

# Define r_i(t), r_f(t), theta_i(t), theta_f(t)
ri = R(i)(t)
rf = R(f)(t)
thetai = Theta(i)(t)
thetaf = Theta(f)(t)

# First derivatives wrt time
dri_dt = sp.diff(ri, t)
dtheta_i_dt = sp.diff(thetai, t)

drf_dt = sp.diff(rf, t)
dtheta_f_dt = sp.diff(thetaf, t)

# Second derivatives wrt time
d2ri_dt2 = sp.diff(ri, (t, 2))
d2theta_i_dt2 = sp.diff(thetai, (t, 2))

d2rf_dt2 = sp.diff(rf, (t, 2))
d2theta_f_dt2 = sp.diff(thetaf, (t, 2))

# Base Lagrangian
L = 0.5 * m * (dri_dt**2 + ri**2 * dtheta_i_dt**2) - m * g * rf * sp.sin(thetaf)
dL_dt = 0.5 * m * ((dri_dt + d2ri_dt2)**2 + (ri+dri_dt)**2 * (dtheta_i_dt+d2theta_i_dt2)**2) - m * g * (rf+drf_dt) * sp.sin(thetaf+dtheta_f_dt)

L + dL_dt = 0



