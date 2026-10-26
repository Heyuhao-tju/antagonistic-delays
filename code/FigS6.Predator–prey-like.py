import argparse
from contextlib import nullcontext
from pathlib import Path
import numpy as np

S, SIGMA, C = 1000, 0.1, 0.1
D_A, D = 0.2, -2.0
RHO, MU = -0.75, 5.0
R_B = SIGMA * np.sqrt(S * C)
TAUS = (0.0, 0.5, 50.0)
SEED = 20261011
B_XLIM, B_YLIM = (-1.6, 2.6), (-2.1, 2.1)
GRID_SIZE = 451
OUTPUT_DIR = Path(__file__).resolve().parent


def generate_b_eigenvalues():
    rng = np.random.default_rng(SEED)
    i, j = np.triu_indices(S, k=1)
    present = rng.random(len(i)) < C
    i, j = i[present], j[present]
    symmetric = rng.standard_normal(len(i))
    antisymmetric = rng.standard_normal(len(i))
    common = np.sqrt((1 + RHO) / 2) * symmetric
    opposite = np.sqrt((1 - RHO) / 2) * antisymmetric
    B = np.zeros((S, S), dtype=float, order='F')
    B[i, j] = SIGMA * (common + opposite)
    B[j, i] = SIGMA * (common - opposite)
    return np.linalg.eigvals(B)


def characteristic_roots(b, tau):
    """Exponential kernels: lambda+d_A-D/(1+mu*lambda)
    -b/(1+tau*lambda)=0. Simplify tau=0 or tau=mu before multiplying.
    """
    b = np.asarray(b, dtype=complex).reshape(-1)
    constant = D_A - D - b
    if tau == 0 or tau == MU:
        linear = (1 + (D_A - b) * MU if tau == 0
                  else np.full(b.shape, 1 + D_A * MU, dtype=complex))
        discriminant = np.sqrt(linear**2 - 4 * MU * constant + 0j)
        return np.column_stack(((-linear + discriminant) / (2 * MU),
                                (-linear - discriminant) / (2 * MU)))
    p2 = (MU + tau + D_A * MU * tau) / (MU * tau)
    p1 = (1 + D_A * (MU + tau) - D * tau - b * MU) / (MU * tau)
    p0 = constant / (MU * tau)
    companion = np.zeros((b.size, 3, 3), dtype=complex)
    companion[:, 0, 0] = -p2
    companion[:, 0, 1] = -p1
    companion[:, 0, 2] = -p0
    companion[:, 1, 0] = 1
    companion[:, 2, 1] = 1
    return np.linalg.eigvals(companion)


def max_real_lambda(b, tau):
    b = np.asarray(b, dtype=complex)
    result = characteristic_roots(b, tau).real.max(axis=1)
    zero = b.reshape(-1) == 0
    if zero.any():
        # At b=0, ignore the artificial interaction pole from multiplication.
        result[zero] = np.roots([MU, 1 + D_A * MU, D_A - D]).real.max()
    return result.reshape(b.shape)


def verify_roots(b, roots, tau):
    self_term = -D / (1 + MU * roots)
    inter_term = b[:, None] / (1 + tau * roots)
    residual = roots + D_A + self_term - inter_term
    scale = 1 + abs(roots) + D_A + abs(self_term) + abs(inter_term)
    assert np.max(abs(residual) / scale) < 1e-9


def critical_curve(omega, tau):
    return (1 + 1j * omega * tau) * (
        1j * omega + D_A - D / (1 + 1j * omega * MU))


def state_name(maximum):
    return 'Unstable' if maximum > 1e-9 else 'Stable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    half_x, half_y = R_B * (1 + RHO), R_B * (1 - RHO)
    theta = np.linspace(0, 2 * np.pi, 20001)
    ellipse = half_x * np.cos(theta) + 1j * half_y * np.sin(theta)
    x, y = (np.linspace(*B_XLIM, GRID_SIZE), np.linspace(*B_YLIM, GRID_SIZE))
    xx, yy = np.meshgrid(x, y)
    inside = (xx / half_x)**2 + (yy / half_y)**2 <= 1
    radius = np.hypot(max(abs(np.array(B_XLIM))), max(abs(np.array(B_YLIM))))
    # Beyond this frequency, |gamma|>=|omega|-|D| exceeds the view radius.
    positive = np.linspace(0, radius + abs(D) + 0.5, 60001)
    omega = np.r_[-positive[:0:-1], positive]

    try:
        from threadpoolctl import threadpool_limits
        context = threadpool_limits(limits=1)
    except ImportError:
        context = nullcontext()
    clouds, grids, states = [], [], []
    with context:
        b = generate_b_eigenvalues()
        for tau in TAUS:
            roots = characteristic_roots(b, tau)
            verify_roots(b, roots, tau)
            selected = roots[np.arange(S), np.argmax(roots.real, axis=1)]
            assert selected.shape == (S,)
            clouds.append(selected)
            states.append(state_name(selected.real.max()))
            ellipse_roots = characteristic_roots(ellipse, tau)
            verify_roots(ellipse, ellipse_roots, tau)
            # Check that ideal support and finite-matrix states agree here.
            assert state_name(ellipse_roots.real.max()) == states[-1]
            grids.append(max_real_lambda(xx + 1j * yy, tau))
        assert states == ['Stable', 'Unstable', 'Stable']

    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.145, top=0.87,
                        wspace=0.23, hspace=0.36)
    stable_root, unstable_root = '#24659B', '#D45A36'
    stable_region, unstable_region = '#F1F6FB', '#ECECEC'
    ellipse_color, critical_color = '#24659B', '#CE652F'
    cmap = ListedColormap([stable_region, unstable_region])

    for column, (tau, points, grid, state) in enumerate(zip(TAUS, clouds, grids, states)):
        # Top row: one rightmost lambda root per actual coupling eigenvalue.
        ax = axes[0, column]
        xmin, xmax = min(points.real.min(), 0), max(points.real.max(), 0)
        span = max(xmax - xmin, 2 * max(abs(points.imag)), 0.001) * 1.15
        xmid = (xmin + xmax) / 2
        xlim = (xmid - span / 2, xmid + span / 2)
        ax.set_xlim(xlim)
        ax.set_ylim(-span / 2, span / 2)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.axvspan(0, xlim[1], color='#EEEEEE', zorder=0)
        ax.axvline(0, color='#444444', lw=0.7)
        ax.axhline(0, color='#CCCCCC', lw=0.5)
        unstable = points.real > 1e-9
        for mask, color in ((~unstable, stable_root), (unstable, unstable_root)):
            ax.scatter(points.real[mask], points.imag[mask], s=5, color=color,
                       alpha=0.75, linewidths=0, rasterized=True)
        ax.set_title(rf'({chr(97+column)}) $\tau={tau:g}$: {state}' + '\n' +
                     rf'$\max\,\mathrm{{Re}}\,\lambda={points.real.max():+.4f}$',
                     pad=9, fontsize=11)
        ax.set_xlabel(r'$\mathrm{Re}\,\lambda$')

        # Bottom row: exact critical curve and identical theoretical ellipse.
        ax = axes[1, column]
        unstable = grid > 0
        ax.pcolormesh(x, y, unstable.astype(float), cmap=cmap, vmin=0, vmax=1,
                      shading='nearest', rasterized=True, zorder=0)
        ax.fill(ellipse.real, ellipse.imag, color=ellipse_color, alpha=0.15, zorder=1)
        if np.any(inside & unstable):
            ax.contourf(x, y, (inside & unstable).astype(float), levels=[0.5, 1.5],
                        colors=['#ECAA92'], alpha=0.75, zorder=2)
        gamma = critical_curve(omega, tau)
        ax.plot(gamma.real, gamma.imag, color=critical_color, lw=1.8, zorder=3)
        ax.plot(ellipse.real, ellipse.imag, color=ellipse_color, lw=1.8, zorder=4)
        ax.scatter([0], [0], s=14, color='#333333', zorder=5)
        ax.axhline(0, color='#888888', lw=0.5, alpha=0.55, zorder=1)
        ax.axvline(0, color='#888888', lw=0.5, alpha=0.55, zorder=1)
        ax.set_xlim(B_XLIM)
        ax.set_ylim(B_YLIM)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_title(rf'({chr(100+column)}) $\tau={tau:g}$: {state}', pad=9, fontsize=11)
        ax.set_xlabel(r'$\mathrm{Re}\,b$')
        for row in range(2):
            axes[row, column].tick_params(labelsize=9)
            axes[row, column].spines['top'].set_visible(False)
            axes[row, column].spines['right'].set_visible(False)

    axes[0, 0].set_ylabel(r'$\mathrm{Im}\,\lambda$')
    axes[1, 0].set_ylabel(r'$\mathrm{Im}\,b$')
    fig.suptitle('Destabilization and restabilization induced by interaction delay'
                 + rf' ($\rho={RHO:g},\ \mu={MU:g}$)', fontsize=13.5, y=0.945)
    handles = [
        Line2D([], [], color=stable_root, marker='o', linestyle='', markersize=4,
               label=r'$\mathrm{Re}\,\lambda\leq0$'),
        Line2D([], [], color=unstable_root, marker='o', linestyle='', markersize=4,
               label=r'$\mathrm{Re}\,\lambda>0$'),
        Line2D([], [], color=ellipse_color, lw=1.8, label='Spectral support ellipse'),
        Line2D([], [], color=critical_color, lw=1.8, label=r'Critical curve $\gamma(\omega)$'),
        Patch(facecolor=stable_region, edgecolor='#B5C8DC', label='Stable region in $b$-plane'),
        Patch(facecolor=unstable_region, edgecolor='#BBBBBB', label='Unstable region in $b$-plane'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.035), fontsize=10, columnspacing=2.4)
    for ext in ('png', 'svg'):
        path = OUTPUT_DIR / f'combined_reentrant_example.{ext}'
        fig.savefig(path, dpi=240, facecolor='white')
        print(f'Saved: {path}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
