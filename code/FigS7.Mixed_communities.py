import argparse
from pathlib import Path

import numpy as np
from scipy.special import lambertw
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits


S, SIGMA, C = 1000, 0.1, 0.1
D_A, D, MU = 1.0, -1.359, 1.0
R_B = SIGMA * np.sqrt(S * C)
TAUS = (0.0, 0.5, 1.0)
SEED = 20261011
OUTPUT_DIR = Path(__file__).resolve().parent
B_XLIM, B_YLIM = (-2.9, 2.9), (-2.4, 2.4)


def generate_b_eigenvalues(rho):
    """Shared reciprocal sparsity; Corr(B_ij, B_ji)=rho."""
    rng = np.random.default_rng(SEED)
    i, j = np.triu_indices(S, k=1)
    present = rng.random(len(i)) < C
    i, j = i[present], j[present]
    zs, za = rng.standard_normal(len(i)), rng.standard_normal(len(i))
    common = np.sqrt((1 + rho) / 2) * zs
    opposite = np.sqrt((1 - rho) / 2) * za
    matrix = np.zeros((S, S), dtype=float)
    matrix[i, j] = SIGMA * (common + opposite)
    matrix[j, i] = SIGMA * (common - opposite)
    return np.linalg.eigvals(matrix)


def equation(lam, b, tau):
    return lam + D_A - D * np.exp(-MU * lam) - b * np.exp(-tau * lam)


def critical_curve(omega, tau):
    return np.exp(1j * tau * omega) * (
        1j * omega + D_A - D * np.exp(-1j * MU * omega))


def select_rightmost(roots, b):
    # Resolve conjugate ties for real b consistently, without duplicating dots.
    score = np.round(roots.real, 10)
    preferred_sign = np.where(b.imag >= 0, 1.0, -1.0)
    score += 1e-12 * np.tanh(roots.imag * preferred_sign[:, None])
    return roots[np.arange(len(b)), score.argmax(axis=1)]


def collocation_roots(b, tau, degree):
    """Generator collocation on [-MU,0]; tau=MU/2 lies on a node."""
    assert tau == MU / 2 and degree % 2 == 0
    x = np.cos(np.pi * np.arange(degree + 1) / degree)
    weights = np.ones(degree + 1)
    weights[[0, -1]] = 2
    weights *= (-1.0) ** np.arange(degree + 1)
    delta = x[:, None] - x[None, :]
    differentiation = (weights[:, None] / weights[None, :]) / (
        delta + np.eye(degree + 1))
    differentiation -= np.diag(differentiation.sum(axis=1))
    differentiation *= 2 / MU
    results = []
    for start in range(0, len(b), 64):
        values = b[start:start + 64]
        matrices = np.broadcast_to(differentiation,
                                  (len(values), degree + 1, degree + 1)).astype(complex).copy()
        matrices[:, 0, :] = 0
        matrices[:, 0, 0] = -D_A
        matrices[:, 0, -1] += D
        matrices[:, 0, degree // 2] += values
        roots = np.linalg.eigvals(matrices)
        indices = np.argsort(roots.real, axis=1)[:, -6:]
        roots = np.take_along_axis(roots, indices, axis=1)
        for _ in range(10):
            self_exp = np.exp(-MU * roots)
            inter_exp = np.exp(-tau * roots)
            residual = roots + D_A - D * self_exp - values[:, None] * inter_exp
            derivative = 1 + D * MU * self_exp + values[:, None] * tau * inter_exp
            correction = residual / derivative
            roots -= correction
            if np.max(abs(correction)) < 2e-13:
                break
        results.append(select_rightmost(roots, values))
    return np.concatenate(results)


def rightmost_roots(b, tau):
    if tau == 0:
        branches = np.arange(-12, 13)
        base = b[:, None] - D_A
        roots = base + lambertw(D * MU * np.exp(-MU * base),
                               branches[None, :]) / MU
        selected = select_rightmost(roots, b)
    elif tau == MU:
        branches = np.arange(-12, 13)
        roots = -D_A + lambertw((D + b[:, None]) * MU * np.exp(D_A * MU),
                                branches[None, :]) / MU
        selected = select_rightmost(roots, b)
    else:
        coarse = collocation_roots(b, tau, 32)
        selected = collocation_roots(b, tau, 48)
        assert np.max(abs(coarse.real - selected.real)) < 1e-8
    scale = 1 + abs(selected) + D_A + abs(D * np.exp(-MU * selected))
    scale += abs(b * np.exp(-tau * selected))
    assert np.max(abs(equation(selected, b, tau)) / scale) < 1e-10
    return selected


def nyquist_counts(b, tau, samples=6001, maximum_b=None):
    """Number of RHP roots using a winding count; requires stable b=0.

    H(z)=exp(-tau*z)/(z+d_A-D*exp(-MU*z)) has no RHP poles here.
    Along the upward imaginary axis, winding of H about 1/b is minus
    the number of unstable roots. The closing RHP arc maps into a disk
    smaller than every |1/b| and may be replaced by a straight segment.
    """
    values = np.asarray(b, dtype=complex)
    flat = values.reshape(-1)
    max_b = float(abs(flat).max()) if maximum_b is None else float(maximum_b)
    cutoff = D_A + abs(D) + max_b + 2.0
    omega = np.linspace(-cutoff, cutoff, samples)
    contour = np.exp(-1j * tau * omega) / (
        1j * omega + D_A - D * np.exp(-1j * MU * omega))
    contour = np.r_[contour, contour[0]]
    nonzero = abs(flat) > 1e-14
    targets = 1 / flat[nonzero]
    active = ((targets.real >= contour.real.min()) & (targets.real <= contour.real.max()) &
              (targets.imag >= contour.imag.min()) & (targets.imag <= contour.imag.max()))
    indices = np.flatnonzero(nonzero)[active]
    px, py = targets.real[active], targets.imag[active]
    winding = np.zeros(len(px), dtype=np.int32)
    for z0, z1 in zip(contour[:-1], contour[1:]):
        y0, y1 = z0.imag, z1.imag
        if y0 == y1:
            continue
        crossed = (py >= min(y0, y1)) & (py < max(y0, y1))
        candidates = np.flatnonzero(crossed)
        if not len(candidates):
            continue
        intersection = z0.real + (py[candidates] - y0) * (z1.real - z0.real) / (y1 - y0)
        hit = candidates[px[candidates] < intersection]
        winding[hit] += 1 if y1 > y0 else -1
    counts = np.zeros(len(flat), dtype=np.int32)
    counts[indices] = -winding
    assert counts.min() >= 0
    return counts.reshape(values.shape)


def ellipse_clearance(rho, tau):
    half_x, half_y = R_B * (1 + rho), R_B * (1 - rho)
    # |gamma(w)| >= w-d_A-|D|: higher frequencies cannot enter the ellipse.
    upper = D_A + abs(D) + max(half_x, half_y) + 1
    omega = np.linspace(0, upper, 20001)
    def gauge(w):
        gamma = critical_curve(w, tau)
        return (gamma.real / half_x) ** 2 + (gamma.imag / half_y) ** 2
    q = gauge(omega)
    candidates = [float(q[0])]
    for i in find_peaks(-q)[0]:
        result = minimize_scalar(gauge, bounds=(omega[i - 1], omega[i + 1]), method='bounded')
        candidates.append(result.fun)
    return np.sqrt(min(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not -1 < args.rho < 1:
        parser.error('rho must be strictly between -1 and 1')
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    with threadpool_limits(limits=1):
        b = generate_b_eigenvalues(args.rho)
        # Stable uncoupled system is needed for the geometric/Nyquist argument.
        origin = -D_A + lambertw(D * MU * np.exp(D_A * MU), np.arange(-12, 13)) / MU
        assert origin.real.max() < 0
        print(f'rho={args.rho:g}; uncoupled max Re(lambda)={origin.real.max():+.9f}', flush=True)
        points = []
        summary = []
        for tau in TAUS:
            roots = rightmost_roots(b, tau)
            counts = nyquist_counts(b, tau)
            refined = nyquist_counts(b, tau, samples=12001)
            assert np.array_equal(counts, refined)
            assert np.array_equal(roots.real > 1e-8, counts > 0)
            state = 'Unstable' if roots.real.max() > 1e-8 else 'Stable'
            clearance = ellipse_clearance(args.rho, tau)
            assert (clearance < 1) == (state == 'Unstable')
            points.append(roots)
            summary.append((tau, roots.real.max(), np.count_nonzero(counts), state, clearance))
            print(f'tau={tau:g}: {state}; max Re(lambda)={roots.real.max():+.9f}; '
                  f'unstable b modes={np.count_nonzero(counts)}', flush=True)

        x = np.linspace(*B_XLIM, 401)
        y = np.linspace(*B_YLIM, 341)
        xx, yy = np.meshgrid(x, y)
        grids = []
        for tau in TAUS:
            grids.append(nyquist_counts(xx + 1j * yy, tau) > 0)
            print(f'Finished b-plane background for tau={tau:g}', flush=True)

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.145, top=0.87,
                        wspace=0.23, hspace=0.36)
    blue, red, orange = '#24659B', '#D45A36', '#CE652F'
    stable_region, unstable_region = '#F1F6FB', '#ECECEC'
    cmap = ListedColormap([stable_region, unstable_region])
    half_x, half_y = R_B * (1 + args.rho), R_B * (1 - args.rho)
    theta = np.linspace(0, 2 * np.pi, 4001)
    ellipse = half_x * np.cos(theta) + 1j * half_y * np.sin(theta)
    inside = (xx / half_x) ** 2 + (yy / half_y) ** 2 <= 1
    radius = np.hypot(max(abs(np.array(B_XLIM))), max(abs(np.array(B_YLIM))))
    omega = np.linspace(-radius - D_A - abs(D) - 1, radius + D_A + abs(D) + 1, 30001)
    # Common upper-row scales make motion across columns directly comparable.
    real_min = min(0, min(z.real.min() for z in points))
    real_max = max(0, max(z.real.max() for z in points))
    span = max(real_max - real_min, 2 * max(abs(z.imag).max() for z in points)) * 1.15
    midpoint = (real_min + real_max) / 2
    lambda_xlim = (midpoint - span / 2, midpoint + span / 2)
    for col, (tau, roots, grid, row) in enumerate(zip(TAUS, points, grids, summary)):
        state = row[3]
        ax = axes[0, col]
        ax.set_xlim(lambda_xlim)
        ax.set_ylim(-span / 2, span / 2)
        ax.axvspan(0, lambda_xlim[1], color=unstable_region, zorder=0)
        ax.axvline(0, color='#444444', lw=0.7)
        ax.axhline(0, color='#CCCCCC', lw=0.5)
        unstable = roots.real > 1e-8
        for mask, color in ((~unstable, blue), (unstable, red)):
            ax.scatter(roots.real[mask], roots.imag[mask], s=5, color=color,
                       alpha=0.8, linewidths=0, rasterized=True)
        ax.set_title(rf'({chr(97 + col)}) $\tau={tau:g}$: {state}' + '\n' +
                     rf'$\max\,\mathrm{{Re}}\,\lambda={roots.real.max():+.4f}$',
                     fontsize=11, pad=9)
        ax.set_xlabel(r'$\mathrm{Re}\,\lambda$')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        ax = axes[1, col]
        ax.pcolormesh(x, y, grid.astype(float), cmap=cmap, vmin=0, vmax=1,
                      shading='nearest', rasterized=True, zorder=0)
        ax.fill(ellipse.real, ellipse.imag, color=blue, alpha=0.15, zorder=1)
        if np.any(inside & grid):
            ax.contourf(x, y, (inside & grid).astype(float), levels=[0.5, 1.5],
                        colors=['#ECAA92'], alpha=0.75, zorder=2)
        curve = critical_curve(omega, tau)
        ax.plot(curve.real, curve.imag, color=orange, lw=1.8, zorder=3)
        ax.plot(ellipse.real, ellipse.imag, color=blue, lw=1.8, zorder=4)
        ax.scatter([0], [0], s=14, color='#333333', zorder=5)
        ax.axhline(0, color='#888888', lw=0.5, alpha=0.55)
        ax.axvline(0, color='#888888', lw=0.5, alpha=0.55)
        ax.set_xlim(B_XLIM)
        ax.set_ylim(B_YLIM)
        ax.set_title(rf'({chr(100 + col)}) $\tau={tau:g}$: {state}', fontsize=11, pad=9)
        ax.set_xlabel(r'$\mathrm{Re}\,b$')
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        for row_index in range(2):
            panel = axes[row_index, col]
            panel.set_aspect('equal', adjustable='box')
            panel.tick_params(labelsize=9)
            panel.spines['top'].set_visible(False)
            panel.spines['right'].set_visible(False)

    axes[0, 0].set_ylabel(r'$\mathrm{Im}\,\lambda$')
    axes[1, 0].set_ylabel(r'$\mathrm{Im}\,b$')
    fig.suptitle('Stabilization and destabilization under discrete interaction delays'
                 + rf' ($\rho={args.rho:g},\ \mu={MU:g}$)', fontsize=13.5, y=0.945)
    handles = [
        Line2D([], [], color=blue, marker='o', linestyle='', markersize=4,
               label=r'$\mathrm{Re}\,\lambda<0$'),
        Line2D([], [], color=red, marker='o', linestyle='', markersize=4,
               label=r'$\mathrm{Re}\,\lambda>0$'),
        Line2D([], [], color=blue, lw=1.8, label='Spectral support ellipse'),
        Line2D([], [], color=orange, lw=1.8, label=r'Critical curve $\gamma(\omega)$'),
        Patch(facecolor=stable_region, edgecolor='#B5C8DC', label='Stable region in $b$-plane'),
        Patch(facecolor=unstable_region, edgecolor='#BBBBBB', label='Unstable region in $b$-plane'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.035), fontsize=10, columnspacing=2.4)
    fig.text(0.5, 0.014,
             rf'$S={S},\ \sigma={SIGMA:g},\ C={C:g},\ d_A={D_A:g},'
             + rf'\ D={D:g},\ r_B={R_B:g}$', ha='center', fontsize=10)
    tag = f'{args.rho:g}'.replace('.', 'p').replace('-', 'm')
    stem = OUTPUT_DIR / f'dirac_nonmonotonic_rho_{tag}'
    for extension in ('png', 'svg'):
        path = stem.with_suffix('.' + extension)
        fig.savefig(path, dpi=240, facecolor='white')
        print(f'Saved: {path}', flush=True)
    np.savez(stem.with_suffix('.npz'), b=b, lambda_rightmost=np.array(points),
             taus=np.array(TAUS), rho=args.rho, d_A=D_A, D=D, mu=MU,
             sigma=SIGMA, C=C, S=S, seed=SEED)
    np.savetxt(str(stem) + '_summary.csv',
               np.array([[r[0], r[1], r[2], r[4]] for r in summary]), delimiter=',',
               header='tau,max_real_lambda,unstable_b_modes,min_ellipse_gauge', comments='')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
