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

# Parameters (Gamma kernel n=2)
d_A = 0.4
D = -1.8
mu_fixed = 10.0
omega_max = 10.0
tau_vals = np.logspace(-1, 3, 1000)   # 0.1 to 1000

# Gamma kernel Laplace transform (shape parameter n=2)
def L_gamma(omega, delay):
    return (1 + 1j * omega * delay / 2.0) ** (-2)

# Compute the critical distance of pure self-delay at mu=10,
# using the scaled variable x = omega * mu
def calc_R_self(mu):
    def dist_x(x):
        omega = x / mu if mu != 0 else 0.0
        if omega < 0 or omega > omega_max:
            return np.inf
        b = 1j * omega + d_A - D * L_gamma(omega, mu)
        return abs(b)

    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]
    res = minimize_scalar(dist_x, bounds=(max(0, x0 - 1), min(50, x0 + 1)),
                          method='bounded')
    min_val = res.fun if res.success else coarse_v[idx]
    val0 = dist_x(0.0)
    return min(val0, min_val)

R_self_val = calc_R_self(mu_fixed)
R_may = d_A - D

# Compute R_inter(tau) for fixed mu, using the scaled variable x = omega * tau
def calc_R_inter(tau):
    def dist_x(x):
        omega = x / tau if tau != 0 else 0.0
        if omega < 0 or omega > omega_max:
            return np.inf
        num = 1j * omega + d_A - D * L_gamma(omega, mu_fixed)
        den = L_gamma(omega, tau)
        b = num / den
        return abs(b)

    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]
    res = minimize_scalar(dist_x, bounds=(max(0, x0 - 1), min(50, x0 + 1)),
                          method='bounded')
    min_val = res.fun if res.success else coarse_v[idx]
    val0 = dist_x(0.0)
    return min(val0, min_val)

R_inter_vals = np.array([calc_R_inter(tau) for tau in tau_vals])

# Plot
fig, ax = plt.subplots(figsize=(8, 4.5))

ax.fill_between(tau_vals, 0, R_self_val, color='white', alpha=1.0)          # S
ax.fill_between(tau_vals, R_self_val, R_may, color='grey', alpha=1)        # Rep
ax.fill_between(tau_vals, R_may, 3.0, color='dimgrey', alpha=1)            # U2

ax.plot(tau_vals, R_inter_vals, 'black', linewidth=2)

ax.axhline(y=R_may, color='black', linestyle='--', linewidth=1)
ax.axhline(y=R_self_val, color='black', linestyle=':', linewidth=1)

ax.set_xscale('log')
ax.set_xlim(tau_vals[0], tau_vals[-1])
ax.set_ylim(0, 3.0)
ax.set_xlabel(r'Interaction delay $\tau$', fontsize=12)
ax.set_ylabel(r'$R_{\mathrm{inter}}(\tau)$', fontsize=12)
ax.set_title(r'Stability recovery by interaction delay ($\mu=10$, $\Gamma$-kernel $n=2$)', fontsize=13)
ax.grid(alpha=0.3, which='major')

regime_cmap = ListedColormap(['white', 'grey', 'dimgrey'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, regime_cmap.N)

sm = mpl.cm.ScalarMappable(cmap=regime_cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2], pad=0.02)
cbar.ax.set_yticklabels(['S', 'Rep', 'U2'], fontsize=9)

plt.tight_layout()
plt.savefig('R_inter_gamma.svg', format='svg', bbox_inches='tight', pad_inches=0.05)
plt.show()