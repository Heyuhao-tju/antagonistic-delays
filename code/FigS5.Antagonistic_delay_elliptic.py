from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np


# Edit experiment parameters here.
S = 1000
SIGMA = 0.1
C = 0.1
D_A = 0.2
D = -2.0
RHOS = (-0.75, -0.5, 0.0, 0.5, 0.75)
SEED = 20261011
STAGES = (
    ("No delay", 0.0, 0.0),
    ("Self-delay", 50.0, 0.0),
    ("Interaction-delay repair", 50.0, 50.0),
)
STABILITY_TOL = 1e-9


def quadratic_roots(A, B, K):
    """Return both roots, with a cancellation-resistant quadratic formula."""
    B, K = np.broadcast_arrays(
        np.asarray(B, dtype=complex), np.asarray(K, dtype=complex)
    )
    discriminant = np.sqrt(B * B - 4.0 * A * K + 0j)
    plus = B + discriminant
    minus = B - discriminant
    q = -0.5 * np.where(np.abs(plus) >= np.abs(minus), plus, minus)
    roots = np.empty((B.size, 2), dtype=complex)
    roots[:, 0] = q / A
    nonzero = np.abs(q) > np.finfo(float).tiny
    np.divide(K, q, out=roots[:, 1], where=nonzero)
    roots[~nonzero, 1] = (-B[~nonzero] + discriminant[~nonzero]) / (2.0 * A)
    return roots


def characteristic_roots(b, mu, tau, a=D_A, D=D):
    """All characteristic roots for exponentially distributed delay kernels.

    lambda + a - D/(1+mu*lambda) - b/(1+tau*lambda) = 0.
    Zero-delay limits and the equal-kernel case are simplified BEFORE
    multiplication, avoiding spurious roots at transform poles.
    """
    b = np.asarray(b, dtype=complex).reshape(-1)
    if mu < 0 or tau < 0:
        raise ValueError("Mean delays must be nonnegative.")
    h = -D
    constant = a + h - b
    if mu == 0 and tau == 0:
        return (-constant)[:, None]
    if tau == 0:
        return quadratic_roots(mu, 1.0 + (a - b) * mu, constant)
    if mu == 0:
        return quadratic_roots(tau, 1.0 + (a + h) * tau, constant)
    if mu == tau:
        # The common denominator gives a quadratic, not a cubic.
        return quadratic_roots(mu, 1.0 + a * mu, constant)
    return np.array(
        [
            np.roots(
                [
                    mu * tau,
                    mu + tau + a * mu * tau,
                    1.0 + a * (mu + tau) + h * tau - bj * mu,
                    a + h - bj,
                ]
            )
            for bj in b
        ]
    )


def rightmost_roots(b, mu, tau):
    """Keep exactly ONE root per b: the one with largest real part.

    In an exact real-part tie, numpy's first maximizer is retained.
    """
    roots = characteristic_roots(b, mu, tau)
    index = np.argmax(roots.real, axis=1)
    selected = roots[np.arange(len(b)), index]
    assert selected.shape == (len(b),)
    return selected


def equation_error(b, roots, mu, tau):
    """Normalized residual in the original rational characteristic equation."""
    b = np.asarray(b)[:, None]
    self_term = -D / (1.0 + mu * roots)
    interaction_term = b / (1.0 + tau * roots)
    residual = roots + D_A + self_term - interaction_term
    scale = 1.0 + np.abs(roots) + D_A + np.abs(self_term) + np.abs(interaction_term)
    return float(np.max(np.abs(residual) / scale))


def shared_random_inputs():
    rng = np.random.default_rng(SEED)
    i, j = np.triu_indices(S, k=1)
    present = rng.random(len(i)) < C
    i, j = i[present], j[present]
    symmetric = rng.standard_normal(len(i))
    antisymmetric = rng.standard_normal(len(i))
    return i, j, symmetric, antisymmetric


def make_matrix(rho, inputs):
    i, j, symmetric, antisymmetric = inputs
    common = np.sqrt((1.0 + rho) / 2.0) * symmetric
    opposite = np.sqrt((1.0 - rho) / 2.0) * antisymmetric
    forward = SIGMA * (common + opposite)
    backward = SIGMA * (common - opposite)
    B = np.zeros((S, S), dtype=float, order="F")
    B[i, j] = forward
    B[j, i] = backward
    empirical_rho = float(np.corrcoef(forward, backward)[0, 1])
    return B, empirical_rho


def state_name(maximum):
    if maximum > STABILITY_TOL:
        return "Unstable"
    if maximum < -STABILITY_TOL:
        return "Stable"
    return "Marginal"


def compute_experiment():
    if S < 2 or SIGMA <= 0 or not (0 < C <= 1) or D_A <= 0 or D >= 0:
        raise ValueError("Require S>=2, sigma>0, 0<C<=1, d_A>0 and D<0.")
    if any(not (-1.0 < rho < 1.0) for rho in RHOS):
        raise ValueError("This experiment uses nondegenerate ellipses: -1<rho<1.")
    try:
        from threadpoolctl import threadpool_limits
        thread_context = threadpool_limits(limits=1)
    except ImportError:
        thread_context = nullcontext()

    inputs = shared_random_inputs()
    b_spectra = np.empty((len(RHOS), S), dtype=complex)
    mapped = np.empty((len(RHOS), len(STAGES), S), dtype=complex)
    rows = []
    with thread_context:
        for row, rho in enumerate(RHOS):
            print(f"Computing {S} eigenvalues for rho={rho:+.2f} ...", flush=True)
            B, measured_rho = make_matrix(rho, inputs)
            b = np.linalg.eigvals(B)
            b_spectra[row] = b
            for column, (name, mu, tau) in enumerate(STAGES):
                all_roots = characteristic_roots(b, mu, tau)
                index = np.argmax(all_roots.real, axis=1)
                selected = all_roots[np.arange(S), index]
                assert selected.size == S
                error = equation_error(b, all_roots, mu, tau)
                if not np.isfinite(error) or error > 1e-8:
                    raise ArithmeticError(f"Characteristic residual too large: {error}")
                mapped[row, column] = selected
                maximum = float(np.max(selected.real))
                item = dict(
                    rho=rho,
                    measured_rho=measured_rho,
                    stage=name,
                    mu=mu,
                    tau=tau,
                    plotted_roots=S,
                    roots_per_b=all_roots.shape[1],
                    max_real_lambda=maximum,
                    unstable_b_count=int(np.count_nonzero(selected.real > STABILITY_TOL)),
                    state=state_name(maximum),
                    normalized_residual=error,
                )
                rows.append(item)
                print(
                    f"  mu={mu:g}, tau={tau:g}: {item['state']}, "
                    f"max Re(lambda)={maximum:+.6f}, {S} plotted roots",
                    flush=True,
                )
    return b_spectra, mapped, rows


def plot_experiment(mapped, output_dir, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        len(STAGES), len(RHOS), figsize=(17.5, 11.5), squeeze=False
    )
    stable_color = "#24659B"
    unstable_color = "#D45A36"

    # Identical limits across rho within each stage, and equal data aspect.
    limits = []
    for row in range(len(STAGES)):
        points = mapped[:, row, :].reshape(-1)
        xmin = min(float(points.real.min()), 0.0)
        xmax = max(float(points.real.max()), 0.0)
        yabs = float(np.max(np.abs(points.imag)))
        span = max(xmax - xmin, 2.0 * yabs, 1e-3) * 1.16
        xmid = (xmin + xmax) / 2.0
        limits.append(((xmid - span / 2, xmid + span / 2), (-span / 2, span / 2)))

    for row, (name, mu, tau) in enumerate(STAGES):
        for column, rho in enumerate(RHOS):
            ax = axes[row, column]
            points = mapped[column, row]
            xlim, ylim = limits[row]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            # The right half-plane is shaded, regardless of point colors.
            ax.axvspan(0.0, xlim[1], color="#EEEEEE", zorder=0)
            ax.axvline(0.0, color="#555555", linewidth=0.8, zorder=1)
            ax.axhline(0.0, color="#CCCCCC", linewidth=0.5, zorder=1)
            unstable = points.real > STABILITY_TOL
            for mask, color in [(~unstable, stable_color), (unstable, unstable_color)]:
                ax.scatter(
                    points.real[mask], points.imag[mask],
                    s=5, color=color, alpha=0.75, linewidths=0,
                    rasterized=True, zorder=2,
                )
            maximum = float(points.real.max())
            ax.text(
                0.04, 0.97,
                f"{state_name(maximum)}\n"
                + r"$\max\,\mathrm{Re}\,\lambda=$"
                + f"{maximum:+.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
            )
            if row == 0:
                ax.set_title(rf"$\rho={rho:g}$", pad=10)
            if column == 0:
                ax.set_ylabel(
                    name + "\n" + rf"$\mu={mu:g},\ \tau={tau:g}$"
                    + "\n" + r"$\mathrm{Im}\,\lambda$"
                )
            if row == len(STAGES) - 1:
                ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
            ax.tick_params(labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    r_B = SIGMA * np.sqrt(S * C)
    fig.suptitle(
        "Rightmost characteristic root for each coupling eigenvalue\n"
        + rf"$S={S},\ \sigma={SIGMA:g},\ C={C:g},\ d_A={D_A:g},"
        + rf"\ D={D:g},\ r_B={r_B:g}$",
        fontsize=15, y=0.987,
    )
    handles = [
        Line2D([], [], color=stable_color, marker="o", linestyle="", markersize=5,
               label=r"$\mathrm{Re}\,\lambda\leq 0$ (within numerical tolerance)"),
        Line2D([], [], color=unstable_color, marker="o", linestyle="", markersize=5,
               label=r"$\mathrm{Re}\,\lambda>0$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.035), fontsize=10)

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.895,
                        wspace=0.24, hspace=0.30)
    for extension in ("png", "svg"):
        target = output_dir / f"elliptic_rightmost_lambda.{extension}"
        fig.savefig(target, dpi=240, facecolor="white")
        print(f"Saved: {target}", flush=True)
    if show:
        plt.show()
    plt.close(fig)


def self_test():
    b = np.array([-0.7, 0.3, 0.2 + 0.8j, 0.2 - 0.8j, -0.4 + 0.6j])
    for mu, tau in [(0, 0), (50, 0), (0, 50), (50, 50), (50, 20)]:
        roots = characteristic_roots(b, mu, tau)
        chosen = rightmost_roots(b, mu, tau)
        assert chosen.size == b.size
        np.testing.assert_allclose(chosen.real, roots.real.max(axis=1))
        assert equation_error(b, roots, mu, tau) < 1e-10
        np.testing.assert_allclose(roots[3], roots[2].conjugate(), atol=1e-10)
    print("Self-tests passed: root selection, original-equation residuals, conjugacy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--show", action="store_true", help="Also show an interactive figure.")
    parser.add_argument("--self-test", action="store_true", help="Check the root solver only.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b_spectra, mapped, rows = compute_experiment()
    metadata = dict(
        S=S, sigma=SIGMA, C=C, d_A=D_A, D=D, r_B=SIGMA*np.sqrt(S*C),
        seed=SEED, rho=list(RHOS), stages=list(STAGES),
        matrix_diagonal=0, reciprocal_connections="shared",
        root_selection="one maximum-real-part root per coupling eigenvalue",
    )
    np.savez_compressed(
        args.output_dir / "elliptic_rightmost_lambda_data.npz",
        b=b_spectra, lambda_rightmost=mapped, rho=np.array(RHOS),
        stages_mu_tau=np.array([(mu, tau) for _, mu, tau in STAGES]),
        metadata=json.dumps(metadata),
    )
    with (args.output_dir / "elliptic_rightmost_lambda_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_experiment(mapped, args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
