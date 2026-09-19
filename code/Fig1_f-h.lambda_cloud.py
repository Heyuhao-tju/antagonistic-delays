import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

d_A = 0.2
D = -2.0
S = 1000
C = 0.1
r_B = 0.5
sigma = r_B / np.sqrt(S * C)

np.random.seed(42)
B_mat = np.zeros((S, S))
mask = np.random.rand(S, S) < C
np.fill_diagonal(mask, False)
B_mat[mask] = np.random.normal(0, sigma, np.sum(mask))
b_eig = np.linalg.eigvals(B_mat)

def compute_lambdas(mu, tau):
    lambdas = []
    for b in b_eig:
        a3 = mu * tau
        a2 = mu + tau + d_A * mu * tau
        a1 = 1 + d_A * (mu + tau) - D * tau - b * mu
        a0 = d_A - D - b
        coeffs = [a0, a1, a2, a3]
        roots = P(coeffs).roots()

        valid = [
            lam for lam in roots
            if abs(1 + lam * mu) > 1e-8 and abs(1 + lam * tau) > 1e-8
        ]
        if not valid:
            continue

        lam_dom = max(valid, key=lambda x: x.real)
        lambdas.append([lam_dom.real, lam_dom.imag])

    return np.array(lambdas)

fig, axes = plt.subplots(3, 1, figsize=(6, 15), sharex=True)

cases = [
    (0, 0, 'No delay: stable'),
    (10, 0, 'Self-delay only: unstable'),
    (10, 10, 'With interaction delay: restored')
]

for i, (ax, (mu, tau, title)) in enumerate(zip(axes, cases)):
    pts = compute_lambdas(mu, tau)

    ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5,
               color='#0346a5', edgecolors='none')

    ax.axvline(x=0, color='black', linewidth=2)

    ylim = max(abs(pts[:, 1]).max(), 3)
    ax.fill_between([-4, 0], -ylim, ylim, color='lightgrey', alpha=0.3)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=12)

    ax.set_ylabel(r'Im($\lambda$)', fontsize=10)
    ax.yaxis.set_label_position("right")

    if i < len(axes) - 1:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r'Re($\lambda$)', fontsize=10)

plt.tight_layout()
plt.savefig('lambda_cloud_vertical_exp.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()