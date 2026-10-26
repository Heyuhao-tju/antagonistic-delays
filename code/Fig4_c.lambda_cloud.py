import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from threadpoolctl import threadpool_limits

# ========== Fixed parameters ==========
d_A = 1.0
D = -0.8
C = 0.1
sigma = 0.1
S = 1000
r_B = sigma * np.sqrt(S * C)

mu = 1.326
tau_values = [0.0, 0.5, 1.0, 1.5, 2.0]

colors = [
    (144/255, 201/255, 231/255),
    (19/255, 103/255, 131/255),
    (0/255, 47/255, 73/255),
    (193/255, 18/255, 33/255),
    (120/255, 0/255, 1/255),
]

# ========== Generate the same random matrix ==========
np.random.seed(42)

mask = np.random.rand(S, S) < C
np.fill_diagonal(mask, False)

B = np.zeros((S, S), dtype=complex)
B[mask] = np.random.normal(0, sigma, size=mask.sum()) + 0j

with threadpool_limits(limits=1):
    z_eig = np.linalg.eigvals(B)

print(
    f"r_B = {r_B:.4f}, "
    f"actual max modulus = {np.max(np.abs(z_eig)):.4f}"
)


# ========== Characteristic equation ==========
def residual(lam, b, mu, tau):
    return (
        lam + d_A
        - D * np.exp(-mu * lam)
        - b * np.exp(-tau * lam)
    )


def select_rightmost(roots, b, mu, tau):
    """Retain one valid rightmost root for every b."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        error = residual(roots, b[:, None], mu, tau)

        scale = (
            1 + abs(roots) + d_A
            + abs(D * np.exp(-mu * roots))
            + abs(b[:, None] * np.exp(-tau * roots))
        )

        valid = np.isfinite(roots) & (abs(error) / scale < 1e-10)

    if not np.all(np.any(valid, axis=1)):
        raise RuntimeError(
            'Root refinement failed; no points were silently omitted.'
        )

    # Resolve conjugate ties consistently.
    sign = np.where(b.imag >= 0, 1.0, -1.0)
    score = np.round(roots.real, 10)
    score += 1e-12 * np.tanh(roots.imag * sign[:, None])
    score[~valid] = -np.inf

    return roots[np.arange(len(b)), np.argmax(score, axis=1)]


def collocation_roots(b, mu, tau, degree):
    """Approximate leading roots and refine them in the original equation."""
    T = max(mu, tau)

    # Chebyshev nodes on the history interval [-T, 0].
    j = np.arange(degree + 1)
    x = np.cos(np.pi * j / degree)

    weights = (-1.0)**j
    weights[[0, -1]] *= 0.5

    delta = x[:, None] - x[None, :]
    differentiation = (
        weights[None, :] / weights[:, None]
    ) / (delta + np.eye(degree + 1))

    differentiation -= np.diag(differentiation.sum(axis=1))
    differentiation *= 2 / T

    # Interpolation allows arbitrary delay locations.
    def history_row(delay):
        target = 1 - 2 * delay / T
        diff = target - x
        index = np.argmin(abs(diff))

        if abs(diff[index]) < 1e-13:
            row = np.zeros(degree + 1)
            row[index] = 1
            return row

        row = weights / diff
        return row / row.sum()

    row_mu = history_row(mu)
    row_tau = history_row(tau)

    results = []

    for start in range(0, len(b), 64):
        values = b[start:start + 64]

        matrices = np.broadcast_to(
            differentiation,
            (len(values), degree + 1, degree + 1)
        ).astype(complex).copy()

        matrices[:, 0, :] = (
            D * row_mu + values[:, None] * row_tau
        )
        matrices[:, 0, 0] -= d_A

        roots = np.linalg.eigvals(matrices)

        # Refine several leading candidates, not just one root.
        indices = np.argsort(roots.real, axis=1)[:, -8:]
        roots = np.take_along_axis(roots, indices, axis=1)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            for _ in range(12):
                e_mu = np.exp(-mu * roots)
                e_tau = np.exp(-tau * roots)

                f = (
                    roots + d_A
                    - D * e_mu
                    - values[:, None] * e_tau
                )
                df = (
                    1 + D * mu * e_mu
                    + values[:, None] * tau * e_tau
                )

                step = f / df
                roots -= step

                if np.max(abs(step)) < 2e-13:
                    break

        results.append(
            select_rightmost(roots, values, mu, tau)
        )

    return np.concatenate(results)


def compute_lambdas(b, mu, tau):
    b = np.asarray(b, dtype=complex)

    if mu == 0 and tau == 0:
        return -d_A + D + b

    with threadpool_limits(limits=1):
        if tau == 0:
            # Exact Lambert-W representation for pure self-delay.
            base = b[:, None] - d_A
            previous = None

            for count in (16, 32):
                branches = np.arange(-count, count + 1)

                roots = base + lambertw(
                    D * mu * np.exp(-mu * base),
                    branches[None, :]
                ) / mu

                current = select_rightmost(roots, b, mu, tau)

                if previous is not None:
                    difference = np.max(
                        abs(current.real - previous.real)
                    )
                    if difference < 1e-9:
                        return current

                previous = current

            raise RuntimeError('Lambert-W branch check failed.')

        # Increase resolution until the rightmost roots converge.
        previous = None

        for degree in (32, 48, 64, 80):
            current = collocation_roots(b, mu, tau, degree)

            if previous is not None:
                difference = np.max(
                    abs(current.real - previous.real)
                )
                if difference < 1e-8:
                    return current

            previous = current

    raise RuntimeError(
        'Increase the collocation degree: roots have not converged.'
    )


# ========== Plot theoretical contour ==========
def plot_contour(ax, mu, tau, r_B, color):
    Lambda_vals = np.linspace(-3.5, 1.0, 1200)
    omega_vals = np.linspace(-2.2, 2.2, 1200)

    Lambda_grid, omega_grid = np.meshgrid(
        Lambda_vals, omega_vals
    )

    exp_mu = np.exp(-mu * Lambda_grid)

    term1 = (
        Lambda_grid**2 + d_A**2 + omega_grid**2
        + 2 * Lambda_grid * d_A
    )
    term2 = D**2 * np.exp(-2 * mu * Lambda_grid)
    term3 = 2 * D * exp_mu * (
        omega_grid * np.sin(omega_grid * mu)
        - np.cos(omega_grid * mu) * (Lambda_grid + d_A)
    )

    mod_sq = term1 + term2 + term3
    F = mod_sq * np.exp(2 * tau * Lambda_grid) - r_B**2

    contour = ax.contour(
        Lambda_grid, omega_grid, F,
        levels=[0],
        colors=[color],
        linewidths=2.5,
        alpha=0.9
    )

    segments = [
        segment for segment in contour.allsegs[0]
        if segment.size > 0
    ]

    if segments:
        points = np.vstack(segments)
        max_idx = np.argmax(points[:, 0])

        rightmost_x = points[max_idx, 0]
        rightmost_y = points[max_idx, 1]

        ax.plot(
            rightmost_x, rightmost_y, 'o',
            color=color, markersize=8,
            markeredgecolor='black'
        )
        ax.plot(
            rightmost_x, -rightmost_y, 'o',
            color=color, markersize=8,
            markeredgecolor='black'
        )


# ========== Plot on the same figure ==========
fig, ax = plt.subplots(figsize=(12, 10))

ax.axvspan(
    0, 1.0,
    color='gray',
    alpha=0.2,
    zorder=0
)

for idx, tau in enumerate(tau_values):
    color = colors[idx]
    print(f"Processing tau = {tau} ...")

    lambdas = compute_lambdas(z_eig, mu, tau)

    assert len(lambdas) == S

    print(
        f"  {len(lambdas)} rightmost roots; "
        f"max Re(lambda) = {lambdas.real.max():+.9f}"
    )

    ax.scatter(
        lambdas.real,
        lambdas.imag,
        color=color,
        alpha=0.2,
        s=12,
        label=f'$\\tau = {tau}$'
    )

    plot_contour(ax, mu, tau, r_B, color)

# ========== Figure formatting ==========
ax.set_xlabel('Re(λ) = Λ', fontsize=16)
ax.set_ylabel('Im(λ) = ω', fontsize=16)

ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim([-2.5, 1.0])
ax.set_ylim([-2.2, 2.2])
ax.set_aspect('equal')

plt.tight_layout()

plt.savefig(
    'lambda_plane_multiple_tau.svg',
    format='svg',
    bbox_inches='tight'
)

plt.show()

