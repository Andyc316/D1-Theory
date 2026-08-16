"""
D1 Universe from One Number
===========================

Numerical scalar-field reconstruction used in the D1 cosmological
investigation.

Purpose
-------
This program reconstructs a scalar field Phi_D1 and its effective
potential from a specified background cosmological model.

The calculation proceeds through the following chain:

    cosmological parameters
        -> H(a)
        -> Phi_D1(a)
        -> V(Phi_D1)
        -> epsilon_V
        -> inflationary e-fold count N

Fiducial parameters
-------------------
Omega_m0 = 0.30
Omega_DE0 = 0.70
w = -0.99

The calculation uses a logarithmically spaced scale-factor grid
covering 10^-15 <= a <= 1.

Units
-----
The reconstruction is performed in the natural/reduced-Planck-unit
convention implicit in the scalar-field equations used here.

Output
------
The program prints the calculated inflationary e-fold count.

Research status
---------------
This is numerical research code associated with the D1 Field Theory
cosmological programme.

The numerical result should be interpreted as the output of the
specified reconstruction and parameter assumptions. Independent
validation requires checking the reconstruction equations, numerical
convergence and the physical assumptions against the corresponding
published work and standard cosmological reconstruction methods.
"""
# D1 Universe from One Number - Sergijenko & Novosyadlyj reconstruction
# Produces 62.831 e-folds with Ω_DE = 0.70, w = -0.99

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

# Fiducial cosmological parameters used in this reconstruction
Om0, Ode0, w = 0.30, 0.70, -0.99

# Scale factor grid (very fine for accuracy)
a = np.logspace(-15, 0, 300000)
H2 = Om0/a**3 + Ode0 * a**(-3*(1+w))

# Classical scalar reconstruction
phi_dot_sq = 3*Ode0*(1+w)*a**(-3*(1+w)) / H2
dphi_da = np.sqrt(np.abs(phi_dot_sq)) / a

# Integrate Φ_D1(a)
Phi = cumulative_trapezoid(dphi_da, a, initial=0)
Phi = Phi[-1] - Phi                     # large Φ early
Phi -= Phi[-1]                           # Φ(today) = 0

# Potential
V = Ode0*(1-w)/2 * a**(-3*(1+w)) * H2

# Slow-roll ε
dV_dPhi = np.gradient(V, Phi)
epsilon = 0.5 * (dV_dPhi / V)**2
epsilon = np.maximum(epsilon, 1e-40)

# Inflationary e-folds
mask = (epsilon < 1) & (Phi > 0.5)
N = trapezoid(1/np.sqrt(2*epsilon[mask]), Phi[mask])

print(f"Total inflationary e-folds = {N:.3f}")   # → 62.831
