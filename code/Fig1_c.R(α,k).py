import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Parameters
d_A = 0.2
D = -2.0
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
    """Compute the global minimum over all nonnegative frequencies."""
    mu = k * alpha
    tau = (1 - k) * alpha

    # No self-delay: the minimum is at zero frequency.
    if mu == 0:
        return R0

    # Pure self-delay: use the exact minimizing frequency.
    if tau == 0:
        S_mu = -2 * D * mu - D * (2 * d_A - D) * mu**2
        omega_sq = max(0.0, (np.sqrt(S_mu) - 1) / mu**2)
        omega = np.sqrt(omega_sq)

        return abs(
            1j * omega + d_A
            - D / (1 + 1j * omega * mu)
        )

    # Set z = omega**2.
    # The derivative of the squared distance gives a cubic in z.
    s = mu**2
    t = tau**2
    b = 1 + d_A**2 * s + 2 * D * mu
    c = R0**2

    coeffs = np.array([
        2 * t * s**2,
        s**2 + s * t * (b + 3),
        2 * (s + t * b),
        b + t * c - s * c,
    ], dtype=float)

    coeffs = np.trim_zeros(coeffs, trim='f')
    roots = np.roots(coeffs / np.max(np.abs(coeffs)))

    # Include zero frequency and every positive real stationary frequency.
    candidates = [0.0]

    for z in roots:
        if z.real > 0 and abs(z.imag) <= 1e-9 * max(1.0, abs(z.real)):
            candidates.append(np.sqrt(z.real))

    values = [
        abs(
            (1 + 1j * omega * tau)
            * (
                1j * omega + d_A
                - D / (1 + 1j * omega * mu)
            )
        )
        for omega in candidates
    ]

    return min(values)


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

# Colorbar with centered ticks
cmap = ListedColormap(colors)
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = BoundaryNorm(bounds, cmap.N)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3, 4], pad=0.02)
cbar.ax.set_yticklabels(
    [f'$k={k}$' for k in k_list],
    fontsize=9,
    rotation=270,
    va='center'
)
cbar.ax.invert_yaxis()

plt.tight_layout()

plt.savefig(
    'R_alpha_family.svg',
    format='svg',
    bbox_inches='tight',
    pad_inches=0.05
)

plt.show()