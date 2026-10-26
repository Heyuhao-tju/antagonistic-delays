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

d_A, D = 0.2, -2.0
omega_max = 10.0
mu_vals = np.logspace(-1, 3, 1000)   # 0.1 to 1000

def R_self(mu):
    if mu == 0:
        return d_A - D   # 2.2

    def dist(omega):
        if omega < 0 or omega > omega_max:
            return np.inf
        b = 1j * omega + d_A - D / (1 + 1j * omega * mu)
        return abs(b)

    # Coarse scan on a log frequency grid to cover very low frequencies
    coarse_w = np.logspace(-4, np.log10(omega_max), 1000)
    coarse_w = np.insert(coarse_w, 0, 0.0)   # explicitly include omega=0
    coarse_v = [dist(w) for w in coarse_w]
    idx = np.argmin(coarse_v)
    w0 = coarse_w[idx]

    # Local refinement
    # Use a relative search radius to adapt to different scales of w0
    if w0 == 0:
        search_width = 1e-3
    else:
        search_width = max(0.5, w0 * 0.5)
    low = max(0.0, w0 - search_width)
    high = min(omega_max, w0 + search_width)
    res = minimize_scalar(dist, bounds=(low, high), method='bounded')

    min_val = res.fun if res.success else coarse_v[idx]
    val0 = dist(0.0)
    return min(min_val, val0)

R_vals = np.array([R_self(mu) for mu in mu_vals])
R_min = d_A                 # 0.2
R_max = d_A - D             # 2.2

fig, ax = plt.subplots(figsize=(8, 4.5))

# Background filled horizontal bands
ax.fill_between(mu_vals, 0, R_min, color='white', alpha=1.0)               # S
ax.fill_between(mu_vals, R_min, R_max, color='grey', alpha=1)             # U1
ax.fill_between(mu_vals, R_max, 3.0, color='dimgrey', alpha=1)           # U2

# Curve
ax.plot(mu_vals, R_vals, 'black', linewidth=2)

# Reference lines
ax.axhline(y=R_max, color='black', linestyle='--', linewidth=1)
ax.axhline(y=R_min, color='black', linestyle=':', linewidth=1)

ax.set_xscale('log')
ax.set_xlim(0.1, 1000)
ax.set_ylim(0, 3.0)
ax.set_xlabel(r'Self-delay $\mu$', fontsize=12)
ax.set_ylabel(r'$R_{\mathrm{self}}(\mu)$', fontsize=12)
ax.set_title(r'Stability under pure self-delay', fontsize=13)
ax.grid(alpha=0.3, which='major')

# Colorbar
regime_cmap = ListedColormap(['white', 'grey', 'dimgrey'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, regime_cmap.N)

sm = mpl.cm.ScalarMappable(cmap=regime_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2], pad=0.02)
cbar.ax.set_yticklabels(['S', 'U1', 'U2'], fontsize=9)

plt.tight_layout()
plt.savefig('R_self_with_colorbar.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()