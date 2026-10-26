import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============ Parameters ============
d_A = 1.0
D = -0.8
mu_fixed = 10.0
tau_vals = np.logspace(-1, 3, 2000)

omega_scan = np.logspace(-4, 3, 50000)
omega_scan = np.insert(omega_scan, 0, 0.0)

def L_gamma(omega, delay, n):
    return (1 + 1j * omega * delay / n) ** (-n)

def L_dirac(omega, delay):
    return np.exp(-1j * omega * delay)

def R_inter(tau, n):
    if n == -1:
        L_self = L_dirac(omega_scan, mu_fixed)
        L_inter = L_dirac(omega_scan, tau)
        numerator = 1j * omega_scan + d_A - D * L_self
        b = numerator / L_inter
        return np.min(np.abs(b))

    if tau <= 100:
        L_self = L_gamma(omega_scan, mu_fixed, n)
        L_inter = L_gamma(omega_scan, tau, n)
        numerator = 1j * omega_scan + d_A - D * L_self
        b = numerator / L_inter
        return np.min(np.abs(b))

    else:
        def dist_x(x):
            omega = x / tau
            if omega < 0 or omega > 1e3:
                return np.inf
            L_self = L_gamma(omega, mu_fixed, n)
            L_inter = L_gamma(omega, tau, n)
            numerator = 1j * omega + d_A - D * L_self
            b = numerator / L_inter
            return np.abs(b)

        coarse_x = np.linspace(0, 20, 400)
        coarse_v = [dist_x(x) for x in coarse_x]
        idx = np.argmin(coarse_v)
        x0 = coarse_x[idx]

        res = minimize_scalar(
            dist_x,
            bounds=(max(0, x0 - 1), min(50, x0 + 1)),
            method='bounded'
        )
        min_val = res.fun if res.success else coarse_v[idx]
        val0 = dist_x(0.0)
        return min(min_val, val0)

n_list_all = [1, 5, 10, 100, -1]

R_inter_curves = []
for n in n_list_all:
    R_inter_curves.append(np.array([R_inter(tau, n) for tau in tau_vals]))

color_specs = [
    (144, 201, 231),
    (19, 103, 131),
    (0, 47, 73),
    (193, 18, 33),
    (120, 0, 1),
]
colors = [tuple(c / 255 for c in rgb) for rgb in color_specs]

plt.figure(figsize=(10, 6))

plt.fill_between(tau_vals, d_A - D, 2.0, color='dimgrey', alpha=0.6)
plt.fill_between(tau_vals, d_A + D, d_A - D, color='lightgrey', alpha=0.6)

labels = [
    r'$\mathrm{Gamma}\ n=1$',
    r'$\mathrm{Gamma}\ n=5$',
    r'$\mathrm{Gamma}\ n=10$',
    r'$\mathrm{Gamma}\ n=100$',
    r'$\mathrm{Dirac}$'
]
for R_vals, col, lab in zip(R_inter_curves, colors, labels):
    plt.plot(tau_vals, R_vals, color=col, linewidth=2, label=lab)

plt.axhline(y=d_A - D, color='black', linestyle='--', linewidth=1.2,
            label=f'May bound ($d_A - D = {d_A - D:.2f}$)')
plt.axhline(y=d_A + D, color='black', linestyle=':', linewidth=1.2,
            label=f'Lower bound ($d_A + D = {d_A + D:.2f}$)')

plt.xscale('log')
plt.xlabel(r'Interaction delay $\tau$')
plt.ylabel(r'$R_{\mathrm{inter}}(\tau)$')
plt.xlim(0.1, 1000)
plt.ylim(0, 2)
plt.grid(alpha=0.3, which='major')

plt.tight_layout()
plt.savefig('R_inter_mollifiers.svg', format='svg', bbox_inches='tight')
plt.show()