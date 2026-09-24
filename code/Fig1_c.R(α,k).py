import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Parameters
d_A = 0.2
D = -2.0
omega_max = 10.0
alpha_vals = np.logspace(-1, 4, 2000)

k_list = [0.0, 0.37, 0.5, 0.7, 1.0]
colors = [
    (144/255, 201/255, 231/255),
    (19/255, 103/255, 131/255),
    (0/255, 47/255, 73/255),
    (193/255, 18/255, 33/255),
    (120/255, 0/255, 1/255),
]

R0 = d_A - D

def calc_R(alpha, k):
    mu = k * alpha
    tau = (1 - k) * alpha

    def dist_x(x):
        omega = x / alpha
        if omega < 0 or omega > omega_max:
            return np.inf
        # critical curve
        b = (1 + 1j * omega * tau) * (1j * omega + d_A - D / (1 + 1j * omega * mu))
        return np.abs(b)

    # coarse grid search in scaled variable x
    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]

    # local refinement
    res = minimize_scalar(
        dist_x,
        bounds=(max(0, x0 - 1), min(50, x0 + 1)),
        method='bounded'
    )
    min_val = res.fun if res.success else coarse_v[idx]

    # explicitly include x=0 (omega=0)
    val0 = dist_x(0.0)
    return min(min_val, val0)

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, col in zip(k_list, colors):
    R_vals = np.array([calc_R(alpha, k) for alpha in alpha_vals])
    ax.plot(alpha_vals, R_vals, color=col, linewidth=2)

ax.set_xscale('log')
ax.set_xlabel(r'Total delay $\alpha = \mu+\tau$', fontsize=12)
ax.set_ylabel(r'$R(\alpha)$', fontsize=12)
ax.set_title(r'$R(\alpha; k)$ for several values of $k$', fontsize=13)
ax.set_xlim(0.1, 10000)
ax.set_ylim(0, 2.5)
ax.grid(alpha=0.3, which='major')

# colorbar with centered ticks
cmap = ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap.N)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3, 4], pad=0.02)
cbar.ax.set_yticklabels([f'$k={k}$' for k in k_list],
                         fontsize=9, rotation=270, va='center')
cbar.ax.invert_yaxis()

plt.tight_layout()
plt.savefig('R_alpha_family.svg', format='svg', bbox_inches='tight', pad_inches=0.05)
plt.show()