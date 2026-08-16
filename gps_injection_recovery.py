"""
GPS Clock Injection and Recovery Test
======================================

Purpose
-------
Numerical injection-and-recovery test for an annual sinusoidal signal
in GPS clock residual data.

The script injects a 200 ps annual sinusoidal signal into GPS clock
residuals, adds Gaussian noise, and fits the injected signal using
nonlinear least-squares optimisation.

Input
-----
Expected input file:

../data/gps_G01_2018_2019_monthly.csv

The input dataset is not currently included in this repository.

Output
------
The script prints the recovered signal amplitude and its fitted
1-sigma uncertainty and produces:

../figures/figure_2_gps_recovery.png

Status
------
Research analysis code associated with the D1 Field Theory programme.

The original input dataset and complete provenance are not currently
archived in this repository. The analysis should therefore be regarded
as research code requiring reconstruction of the original data pipeline
before independent reproduction of the published result.

"""
# GPS clock injection & recovery test
import numpy as np, matplotlib.pyplot as plt
import pandas as pd

# Load real GPS data (you already have this)
data = pd.read_csv('../data/gps_G01_2018_2019_monthly.csv')
t = np.arange(len(data))
residual = data['clock_ps'].values

# Inject 200 ps annual D1 signal
phase = 2*np.pi*(t - 80)/365.25
injected = 200 * np.sin(phase)
np.random.normal(0, 20, len(t))

# Fit
from scipy.optimize import curve_fit
def model(t, A): return A * np.sin(2*np.pi*(t-80)/365.25)
popt, pcov = curve_fit(model, t, noisy, p0=[190])
print(f"Recovered amplitude = {popt[0]:.1f} ± {np.sqrt(pcov[0,0]):.1f} ps")

# Plot (your exact Figure 2)
plt.figure(figsize=(8,4))
plt.scatter(t, noisy, c='steelblue', s=20, label='Real GPS + injected signal')
plt.plot(t, model(t, *popt), 'r-', lw=2, label=f'Fit: {popt[0]:.1f} ps')
plt.xlabel('Days since 2018-01-01')
plt.ylabel('Clock residual (ps)')
plt.legend()
plt.tight_layout()
plt.savefig('../figures/figure_2_gps_recovery.png', dpi=300)
