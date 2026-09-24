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

# Gamma kernel parameters (shape n=2)
d_A = 0.4
D = -1.8
k = 0.7
omega_max = 10.0
alpha_vals = np.logspace(-1, 4, 2000)

def L_gamma(omega, delay):
    return (1 + 1j * omega * delay / 2.0) ** (-2)

def calc_R(alpha):
    mu = k * alpha
    tau = (1 - k) * alpha

    def dist_x(x):
        omega = x / alpha
        if omega < 0 or omega > omega_max:
            return np.inf
        L_mu = L_gamma(omega, mu)
        L_tau = L_gamma(omega, tau)
        b = (1j * omega + d_A - D * L_mu) / L_tau
        return np.abs(b)

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

def calc_R_infty(k):
    def F_inf(x):
        A = (1 + 1j * (1 - k) * x / 2) ** 2
        B = d_A - D * (1 + 1j * k * x / 2) ** (-2)
        return abs(A * B)

    xs = np.linspace(0, 20, 2000)
    ys = [F_inf(x) for x in xs]
    idx = np.argmin(ys)
    x0 = xs[idx]

    res = minimize_scalar(F_inf, bounds=(max(0, x0 - 1), min(50, x0 + 1)),
                          method='bounded')
    return res.fun if res.success else ys[idx]

R_vals = np.array([calc_R(alpha) for alpha in alpha_vals])
R_star = np.min(R_vals)
R_inf = calc_R_infty(k)
R0 = d_A - D

print(f"R_star = {R_star:.4f}")
print(f"R_inf  = {R_inf:.4f}")
print(f"R0     = {R0:.4f}")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.fill_between(alpha_vals, 0, R_star, color='white', alpha=1.0)
ax.fill_between(alpha_vals, R_star, R_inf, color='lightgrey', alpha=1.0)
ax.fill_between(alpha_vals, R_inf, R0, color='grey', alpha=1.0)
ax.fill_between(alpha_vals, R0, 3.5, color='dimgrey', alpha=1.0)

ax.plot(alpha_vals, R_vals, 'black', linewidth=2)

ax.axhline(y=R0, color='black', linestyle='--', linewidth=1)
ax.axhline(y=R_inf, color='grey', linestyle=':', linewidth=1)
ax.axhline(y=R_star, color='black', linestyle='-.', linewidth=1)

ax.set_xscale('log')
ax.set_xlim(alpha_vals[0], alpha_vals[-1])
ax.set_ylim(0, 2.5)
ax.set_xlabel(r'Total delay $\alpha$', fontsize=12)
ax.set_ylabel(r'$R(\alpha)$', fontsize=12)
ax.set_title(r'Stability regions for $k=0.7$ ($\Gamma$-kernel $n=2$)', fontsize=13)
ax.grid(alpha=0.3, which='major')

regime_cmap = ListedColormap(['white', 'lightgrey', 'grey', 'dimgrey'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, regime_cmap.N)

sm = mpl.cm.ScalarMappable(cmap=regime_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3], pad=0.02)
cbar.ax.set_yticklabels(['S', 'R', 'U1', 'U2'], fontsize=9)

plt.tight_layout()
plt.savefig('k07_regimes_gamma.svg', format='svg', bbox_inches='tight', pad_inches=0.05)
plt.show()