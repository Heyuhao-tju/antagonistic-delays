import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Parameters
d_A = 1.0
D = -0.8
mu_vals = np.logspace(-1, 3, 2000)

n_list = [1, 5, 10, 100]

def L_gamma(omega, mu, n):
    return (1 + 1j * omega * mu / n) ** (-n)

def L_dirac(omega, mu):
    return np.exp(-1j * omega * mu)

def R_self(mu, L_func):
    def dist_x(x):
        omega = x / mu
        if omega < 0 or omega > 1e3:
            return np.inf
        b = 1j * omega + d_A - D * L_func(omega, mu)
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

R_curves = []
for n in n_list:
    R_curves.append(np.array([R_self(mu, lambda w, m, n=n: L_gamma(w, m, n)) for mu in mu_vals]))
R_dirac = np.array([R_self(mu, L_dirac) for mu in mu_vals])

color_specs = [
    (144, 201, 231),
    (19, 103, 131),
    (0, 47, 73),
    (193, 18, 33),
    (120, 0, 1),
]
colors = [tuple(c / 255 for c in rgb) for rgb in color_specs]

plt.figure(figsize=(10, 6))

plt.fill_between(mu_vals, d_A - D, 2.0, color='dimgrey', alpha=0.6)
plt.fill_between(mu_vals, d_A + D, d_A - D, color='lightgrey', alpha=0.6)

labels = [r'$\mathrm{Gamma}\ n=1$', r'$\mathrm{Gamma}\ n=5$',
          r'$\mathrm{Gamma}\ n=10$', r'$\mathrm{Gamma}\ n=100$', r'$\mathrm{Dirac}$']
all_curves = R_curves + [R_dirac]
for R_vals, col, lab in zip(all_curves, colors, labels):
    plt.plot(mu_vals, R_vals, color=col, linewidth=2, label=lab)

plt.axhline(y=d_A - D, color='black', linestyle='--', linewidth=1.2,
            label=f'May bound ($d_A - D = {d_A - D:.2f}$)')
plt.axhline(y=d_A + D, color='black', linestyle=':', linewidth=1.2,
            label=f'Lower bound ($d_A + D = {d_A + D:.2f}$)')

plt.xscale('log')
plt.xlabel(r'Self-delay $\mu$')
plt.ylabel(r'$R_{\mathrm{self}}(\mu)$')
plt.xlim(0.1, 1000)
plt.ylim(0, 2)
plt.grid(alpha=0.3, which='major')

plt.tight_layout()
plt.savefig('R_self_mollifiers.svg', format='svg', bbox_inches='tight')
plt.show()