import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Gamma kernel parameters (shape n=2)
d_A = 0.4
D = -1.8
omega_max = 10.0
mu_vals = np.logspace(-1, 3, 1000)

def L_mu(omega, mu):
    return (1 + 1j * omega * mu / 2.0) ** (-2)

def R_self(mu):
    def dist_x(x):
        omega = x / mu if mu != 0 else 0.0
        if omega < 0:
            return np.inf
        b = 1j * omega + d_A - D * L_mu(omega, mu)
        return abs(b)

    coarse_x = np.linspace(0, 20, 1000)
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

R_vals = np.array([R_self(mu) for mu in mu_vals])

R_star = np.min(R_vals)

def compute_R_infty_gamma():
    def g(x):
        term = (1 + 1j * x / 2.0) ** (-2)
        val = d_A - D * term
        return np.abs(val)

    x_vals = np.linspace(0, 20, 2000)
    y_vals = [g(x) for x in x_vals]
    idx = np.argmin(y_vals)
    x0 = x_vals[idx]

    res = minimize_scalar(g, bounds=(max(0, x0 - 1), min(50, x0 + 1)),
                          method='bounded')
    return res.fun if res.success else y_vals[idx]

R_inf = compute_R_infty_gamma()
R0 = d_A - D

print(f"R_star = {R_star:.4f}, d_A = {d_A}, R0 = {R0:.4f}, R_inf = {R_inf:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.fill_between(mu_vals, 0, R_star, color='white', alpha=1.0)
ax.fill_between(mu_vals, R_star, R_inf, color='lightgrey', alpha=1.0)
ax.fill_between(mu_vals, R_inf, R0, color='grey', alpha=1.0)
ax.fill_between(mu_vals, R0, 3.0, color='dimgrey', alpha=1.0)

ax.plot(mu_vals, R_vals, 'black', linewidth=2)

ax.axhline(y=R0, color='black', linestyle='--', linewidth=1)
ax.axhline(y=R_inf, color='grey', linestyle=':', linewidth=1)
ax.axhline(y=R_star, color='black', linestyle='-.', linewidth=1)

ax.set_xscale('log')
ax.set_xlim(0.1, 1000)
ax.set_ylim(0, 3.0)
ax.set_xlabel(r'Self-delay $\mu$', fontsize=12)
ax.set_ylabel(r'$R_{\mathrm{self}}(\mu)$', fontsize=12)
ax.set_title(r'Stability under pure self-delay ($\Gamma$-kernel, $n=2$)', fontsize=13)
ax.grid(alpha=0.3, which='major')

regime_cmap = ListedColormap(['white', 'lightgrey', 'grey', 'dimgrey'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, regime_cmap.N)

sm = mpl.cm.ScalarMappable(cmap=regime_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3], pad=0.02)
cbar.ax.set_yticklabels(['S', 'R', 'U1', 'U2'], fontsize=9)

plt.tight_layout()
plt.savefig('R_self_gamma.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()