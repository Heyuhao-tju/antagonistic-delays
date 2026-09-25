import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

# Global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Parameters
d_A = 0.2
D = -2.0
k = 0.7
omega_max = 10.0
alpha_vals = np.logspace(-1, 4, 2000)

# Compute R(alpha) using scaled variable x = omega * alpha
def calc_R(alpha):
    mu = k * alpha
    tau = (1 - k) * alpha

    def dist_x(x):
        omega = x / alpha
        if omega < 0 or omega > omega_max:
            return np.inf
        b = (1 + 1j * omega * tau) * (1j * omega + d_A - D / (1 + 1j * omega * mu))
        return np.abs(b)

    # Coarse scan over scaled variable x
    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]

    # Local refinement
    res = minimize_scalar(
        dist_x,
        bounds=(max(0, x0 - 1), min(50, x0 + 1)),
        method='bounded'
    )
    min_val = res.fun if res.success else coarse_v[idx]

    val0 = dist_x(0.0)   # corresponds to omega = 0
    return min(min_val, val0)

R_vals = np.array([calc_R(alpha) for alpha in alpha_vals])
R_star = np.min(R_vals)

# Compute R_infty(k) using the limit formula
C = (-D) * (2 * d_A + (-D))

def R_infty(k):
    def f(x):
        F = (1 + (1 - k)**2 * x**2) * (d_A**2 + C / (1 + k**2 * x**2))
        return np.sqrt(F)

    xs = np.linspace(0, 20, 2000)
    fs = f(xs)
    x0 = xs[np.argmin(fs)]
    res = minimize_scalar(f, bounds=(max(0, x0 - 2), x0 + 2), method='bounded')
    return res.fun if res.success else np.min(fs)

R_inf = R_infty(k)
R0 = d_A - D

# Plot background bands
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(alpha_vals, 0, R_star, color='white', alpha=1.0)
ax.fill_between(alpha_vals, R_star, R_inf, color='lightgrey', alpha=1.0)
ax.fill_between(alpha_vals, R_inf, R0, color='grey', alpha=1.0)
ax.fill_between(alpha_vals, R0, 3.5, color='dimgrey', alpha=1.0)

# Curve
ax.plot(alpha_vals, R_vals, 'black', linewidth=2)

# Reference lines
ax.axhline(y=R0, color='black', linestyle='--', linewidth=1)
ax.axhline(y=R_inf, color='grey', linestyle=':', linewidth=1)
ax.axhline(y=R_star, color='black', linestyle='-.', linewidth=1)

ax.set_xscale('log')
ax.set_xlim(0.1, 10000)
ax.set_ylim(0, 2.5)
ax.set_xlabel(r'Total delay $\alpha$', fontsize=12)
ax.set_ylabel(r'$R(\alpha)$', fontsize=12)
ax.set_title(r'Stability regions for $k=0.7$', fontsize=13)
ax.grid(alpha=0.3, which='major')

# Add colorbar
regime_cmap = ListedColormap(['white', 'lightgrey', 'grey', 'dimgrey'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, regime_cmap.N)

sm = mpl.cm.ScalarMappable(cmap=regime_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3], pad=0.02)
cbar.ax.set_yticklabels(['S', 'R', 'U1', 'U2'], fontsize=9)

plt.tight_layout()
plt.savefig('k07_regimes.svg', format='svg', bbox_inches='tight', pad_inches=0.05)
plt.show()