import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

# Parameters (Gamma kernel n=2)
d_A = 0.4
D = -1.8
S = 1000
C = 0.1
r_B = 1
sigma = r_B / np.sqrt(S * C)

# Generate random matrix eigenvalues (fixed seed)
np.random.seed(42)
B_mat = np.zeros((S, S))
mask = np.random.rand(S, S) < C
np.fill_diagonal(mask, False)
B_mat[mask] = np.random.normal(0, sigma, np.sum(mask))
b_eig = np.linalg.eigvals(B_mat)

def compute_lambdas(mu, tau):
    """
    Solve the characteristic equation for Gamma kernel n=2:
        λ = -d_A + D*(1+λμ/2)^(-2) + b*(1+λτ/2)^(-2)
    """
    lambdas = []
    for b in b_eig:
        A = P([1, mu/2])       # 1 + (μ/2)λ
        B = P([1, tau/2])
        A2 = A * A
        B2 = B * B
        L_plus_dA = P([d_A, 1])
        term1 = L_plus_dA * A2 * B2
        poly = term1 - D * B2 - b * A2
        roots = poly.roots()
        valid = [lam for lam in roots
                 if abs(1 + lam*mu/2) > 1e-8 and abs(1 + lam*tau/2) > 1e-8]
        if not valid:
            continue
        lam_dom = max(valid, key=lambda x: x.real)
        lambdas.append([lam_dom.real, lam_dom.imag])
    return np.array(lambdas)

fig, axes = plt.subplots(3, 1, figsize=(6, 15))

cases = [
    (0, 0, 'No delay: stable',                    (-3, 3),  (-3, 3)),
    (5, 0, 'Self‑delay only: unstable',           (-1.5, 1.5),  (-1.5, 1.5)),
    (5, 10, 'With interaction delay: restored',   (-0.75, 0.75),  (-0.75, 0.75))
]

for i, (ax, (mu, tau, title, xlim, ylim)) in enumerate(zip(axes, cases)):
    pts = compute_lambdas(mu, tau)
    ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.5,
               color='#0346a5', edgecolors='none')
    ax.axvline(x=0, color='black', linewidth=2)

    ylim_abs = max(abs(ylim[0]), abs(ylim[1]))
    ax.fill_between([xlim[0], 0], ylim[0], ylim[1],
                    color='lightgrey', alpha=0.3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=12)

    ax.set_ylabel(r'Im($\lambda$)', fontsize=10)
    ax.yaxis.set_label_position("right")

    ax.tick_params(labelbottom=True)

    if i == len(axes) - 1:
        ax.set_xlabel(r'Re($\lambda$)', fontsize=10)

fig.tight_layout(h_pad=1.6)
plt.savefig('lambda_cloud_vertical_gamma.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()