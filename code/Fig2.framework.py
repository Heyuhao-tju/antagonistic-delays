from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import proj3d
import numpy as np


D_A, D, R_B = 0.2, -2.0, 1.0
R_0 = D_A - D
MAX_DELAY = 120.0
MU_FIXED, TAU_FINAL = 50.0, 75.0
OUT = Path(__file__).resolve().parent

INK = "#172E42"
MUTED = "#5C7080"
GRID = "#D8E3E9"
SELF = "#D05A32"
INTER = "#167AA7"
JIRSA = "#009E73"
PIGANI = "#77519D"
STABLE = "#1C806D"
BG = "#FFFFFF"


def squared_self_distance(x, mu):
    """A_mu at squared frequency x."""
    h = -D
    s = mu * mu
    b = 1 + D_A * D_A * s - 2 * h * mu
    return (s * x * x + b * x + R_0 * R_0) / (1 + s * x)


def threshold(mu, tau, return_frequency=False):
    """Evaluate the global minimum by checking every stationary frequency."""
    if mu == 0:
        return (R_0, 0.0) if return_frequency else R_0
    s, t = mu * mu, tau * tau
    b, c = 1 + D_A * D_A * s + 2 * D * mu, R_0 * R_0
    coefficients = np.array([
        2 * t * s * s,
        s * s + s * t * (b + 3),
        2 * (s + t * b),
        b + t * c - s * c,
    ], dtype=float)
    coefficients = np.trim_zeros(coefficients, trim="f")
    roots = np.roots(coefficients / np.max(np.abs(coefficients)))
    candidates = [0.0]
    candidates += [z.real for z in roots
                   if z.real > 0 and abs(z.imag) <= 1e-8 * max(1, abs(z.real))]
    values = [(1 + t * x) * squared_self_distance(x, mu) for x in candidates]
    j = int(np.argmin(values))
    result = np.sqrt(max(values[j], 0.0))
    return (result, np.sqrt(candidates[j])) if return_frequency else result


def validate():
    # Independent closed-form checks on the self-delay and equal-delay slices.
    for mu in np.geomspace(0.005, MAX_DELAY, 80):
        h = -D
        q = 2 * h * mu + h * (2 * D_A + h) * mu * mu
        x_self = max(0.0, (np.sqrt(q) - 1) / (mu * mu))
        expected_self = np.sqrt(squared_self_distance(x_self, mu))
        assert abs(threshold(mu, 0) - expected_self) < 1e-8
        b = 1 + D_A * D_A * mu * mu + 2 * D * mu
        expected_equal = np.sqrt(R_0 * R_0 - min(b, 0.0)**2 / (4 * mu * mu))
        assert abs(threshold(mu, mu) - expected_equal) < 1e-8
    for mu in (0, 0.1, 1, 5, 50, 120):
        values = np.array([threshold(mu, t) for t in np.linspace(0, MAX_DELAY, 100)])
        assert np.min(np.diff(values)) > -1e-8
        assert np.max(values) <= R_0 + 1e-8
    assert threshold(0, 0) > R_B
    assert threshold(MU_FIXED, 0) < R_B
    assert threshold(MU_FIXED, TAU_FINAL) > R_B


def callout(ax, xyz, text, offset, color=INK, align="left", fontsize=11.5,
            figure_position=None, leader=True):
    px, py, _ = proj3d.proj_transform(*xyz, ax.get_proj())
    return ax.annotate(
        text, xy=(px, py),
        xytext=offset if figure_position is None else figure_position,
        textcoords="offset points" if figure_position is None else "figure fraction",
        ha=align, va="center", fontsize=fontsize, color=color,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.28", fc=BG, ec="none", alpha=0.94),
        arrowprops=(dict(arrowstyle="-", color=color, lw=0.8,
                         shrinkA=4, shrinkB=6,
                         connectionstyle="angle3,angleA=0,angleB=80")
                    if leader else None),
        zorder=30,
    )


def arrow3d(ax, start, end, color, lw=2.6):
    xs, ys = [], []
    for xyz in (start, end):
        xp, yp, _ = proj3d.proj_transform(*xyz, ax.get_proj())
        xs.append(xp)
        ys.append(yp)
    ann = ax.annotate(
        "", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14, shrinkA=0, shrinkB=0),
        zorder=25,
    )
    return ann


def main():
    validate()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "mathtext.fontset": "dejavusans", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": BG,
    })
    fig = plt.figure(figsize=(12.6, 7.6), facecolor=BG)
    title = fig.text(0.045, 0.955, "A unified landscape of delay effects",
                     fontsize=18, weight="medium", color=INK)

    # (a) True stability-threshold surface.
    ax = fig.add_axes([0.005, 0.105, 0.70, 0.76], projection="3d", computed_zorder=False)
    grid = MAX_DELAY * np.linspace(0, 1, 145)**2
    mu_grid, tau_grid = np.meshgrid(grid, grid)
    rr = np.empty_like(mu_grid)
    for i, tau in enumerate(grid):
        for j, mu in enumerate(grid):
            rr[i, j] = threshold(mu, tau) / R_0
    cmap = LinearSegmentedColormap.from_list(
        "stability_landscape",
        ["#D57B32", "#F4C474", "#F0EDC6", "#75B4C4", "#205277"])
    colors = cmap(Normalize(0.08, 1)(rr))
    colors[..., 3] = 1.0
    ax.plot_surface(mu_grid, tau_grid, rr, facecolors=colors,
                    rstride=2, cstride=2, linewidth=0, edgecolor="none",
                    antialiased=False,
                    shade=True, zorder=1)
    ax.set(xlim=(0, MAX_DELAY), ylim=(0, MAX_DELAY), zlim=(0, 1.08))
    ax.view_init(elev=28, azim=-110)
    ax.set_box_aspect((1.25, 1.25, 0.88))
    ax.set_xlabel(r"$\mu$", labelpad=10, fontsize=12.5)
    ax.set_ylabel(r"$\tau$", labelpad=11, fontsize=12.5)
    ax.set_zlabel(r"$R/R_0$", labelpad=8, fontsize=12)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_yticks([0, 40, 80, 120])
    ax.set_zticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=9.5, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
        axis.line.set_color(GRID)

    ts = MAX_DELAY * np.linspace(0, 1, 400)**2
    halo = [pe.Stroke(linewidth=5.2, foreground=BG, alpha=0.95), pe.Normal()]
    # Jirsa-type mu=0 edge and common exponential-kernel Pigani slice.
    line, = ax.plot(np.zeros_like(ts), ts, np.ones_like(ts),
                    color=JIRSA, lw=2.7, zorder=9)
    line.set_path_effects(halo)
    pigani_r = np.array([threshold(t, t) for t in ts]) / R_0
    # A thin Pigani line sits above the wider AB line so both colors remain
    # visible where their projections overlap, without shifting either curve.
    line, = ax.plot(ts, ts, pigani_r, color=PIGANI, lw=2.2, zorder=12.5)
    line.set_path_effects([
        pe.Stroke(linewidth=3.0, foreground=BG, alpha=0.95), pe.Normal()
    ])
    ax.plot(ts, ts, np.zeros_like(ts), color=PIGANI, alpha=0.18,
            lw=1.1, zorder=0)

    # Independent two-stage trajectory.
    mu_path = np.expm1(np.linspace(0, np.log1p(MU_FIXED), 260))
    tau_path = np.linspace(0, TAU_FINAL, 260)
    self_r = np.array([threshold(mu, 0) for mu in mu_path]) / R_0
    repair_r = np.array([threshold(MU_FIXED, t) for t in tau_path]) / R_0
    line, = ax.plot(mu_path, np.zeros_like(mu_path), self_r,
                    color=SELF, lw=4.8, zorder=12)
    line.set_path_effects([
        pe.Stroke(linewidth=6.4, foreground=BG, alpha=0.95), pe.Normal()
    ])
    line, = ax.plot(np.full_like(tau_path, MU_FIXED), tau_path, repair_r,
                    color=INTER, lw=3.0, zorder=13)
    line.set_path_effects(halo)
    points = [(0, 0, 1.0), (MU_FIXED, 0, self_r[-1]),
              (MU_FIXED, TAU_FINAL, repair_r[-1])]
    for xyz, col in zip(points, (INK, SELF, INTER)):
        ax.scatter(*xyz, s=57, color=col, edgecolor=BG, linewidth=1.5,
                   depthshade=False, zorder=20)

    # Align the heading with the active 3D box rather than its wider slot.
    title.set_x(ax.get_position().x0 - 0.015)

    # (b) Mechanistic distinction: a delayed self-loop versus delayed links.
    network = fig.add_axes([0.640, 0.565, 0.245, 0.285])
    network.set(xlim=(-0.22, 1.22), ylim=(0, 1.12), aspect="equal")
    network.axis("off")
    xi, xj, yy, radius = 0.13, 0.91, 0.39, 0.12
    loop = MplPath([(xi-0.075, yy+0.09), (xi-0.48, yy+0.72),
                    (xi+0.46, yy+0.72), (xi+0.075, yy+0.09)],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    network.add_patch(FancyArrowPatch(path=loop, arrowstyle="-|>",
                                     mutation_scale=12, lw=2.3, color=SELF))
    for p1, p2 in [((xi+radius, yy+0.025), (xj-radius, yy+0.025)),
                   ((xj-radius, yy-0.025), (xi+radius, yy-0.025))]:
        network.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                         connectionstyle="arc3,rad=-0.25",
                                         mutation_scale=12, lw=2.1, color=INTER))
    for x, label in ((xi, "$i$"), (xj, "$j$")):
        network.add_patch(Circle((x, yy), radius, facecolor=BG,
                                 edgecolor=INK, lw=1.5, zorder=8))
        network.text(x, yy, label, ha="center", va="center", fontsize=16, color=INK, zorder=9)
    network.text(xi, 0.99, r"$\mu$", ha="center", fontsize=13, color=SELF)
    network.text(0.52, 0.20, r"$\tau$", ha="center", fontsize=13, color=INTER)
    fig.text(0.625, 0.867, "b", fontsize=15, weight="bold")

    # (c) The same trajectory as on the surface, now showing the stability test.
    curve = fig.add_axes([0.657, 0.205, 0.215, 0.268])
    xx = np.linspace(0, 1, len(mu_path))
    yy2 = np.linspace(1, 2, len(tau_path))
    x_all = np.r_[xx, yy2[1:]]
    r_all = np.r_[self_r, repair_r[1:]]
    radius_normalized = R_B / R_0
    curve.fill_between(x_all, radius_normalized, r_all, where=r_all >= radius_normalized,
                       color=STABLE, alpha=0.085, interpolate=True)
    curve.fill_between(x_all, radius_normalized, r_all, where=r_all < radius_normalized,
                       color=SELF, alpha=0.10, interpolate=True)
    curve.axhline(radius_normalized, color=MUTED, lw=1, ls=(0, (4, 3)))
    curve.axvline(1, color=GRID, lw=0.8, zorder=0)
    curve.plot(xx, self_r, color=SELF, lw=2.5)
    curve.plot(yy2, repair_r, color=INTER, lw=2.5)
    for x, y, color in ((0, 1, INK), (1, self_r[-1], SELF), (2, repair_r[-1], INTER)):
        curve.scatter(x, y, color=color, edgecolor=BG, s=37, lw=1.0, zorder=5, clip_on=False)
    curve.text(0.05, 1.065, "Stable", fontsize=9.5, color=STABLE)
    curve.text(1.95, 1.065, "Stable", ha="right", fontsize=9.5, color=STABLE)
    curve.text(1.0, 0.025, "Unstable", ha="center", fontsize=9.5, color=SELF)
    curve.text(1.93, radius_normalized+0.035, r"$r_B/R_0$", ha="right", fontsize=10, color=MUTED)
    curve.set(xlim=(-0.05, 2.05), ylim=(0, 1.16), yticks=(0, 0.5, 1),
              xticks=(0, 1, 2), xticklabels=("A", "B", "C"))
    curve.set_ylabel(r"$R/R_0$", fontsize=11, labelpad=3)
    curve.tick_params(labelsize=9, length=3)
    curve.spines[["top", "right"]].set_visible(False)
    curve.annotate("", xy=(0.46, -0.20), xytext=(0.04, -0.20), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", lw=1.3, color=SELF))
    curve.annotate("", xy=(0.96, -0.20), xytext=(0.55, -0.20), xycoords="axes fraction",
                   arrowprops=dict(arrowstyle="->", lw=1.3, color=INTER))
    curve.text(0.25, -0.34, r"$\mu\uparrow$", transform=curve.transAxes,
               ha="center", fontsize=10, color=SELF)
    curve.text(0.75, -0.34, r"$\tau\uparrow$", transform=curve.transAxes,
               ha="center", fontsize=10, color=INTER)
    fig.text(0.625, 0.505, "c", fontsize=15, weight="bold")

    fig.add_artist(plt.Line2D([0.610, 0.610], [0.12, 0.865],
                              transform=fig.transFigure, color=GRID, lw=0.8))

    fig.canvas.draw()
    # Anchor the panel letter beside the visible surface, not the empty slot.
    label_a = callout(ax, (0, MAX_DELAY, 1), "a", (-12, 32), INK,
                      align="right", fontsize=15, leader=False)
    label_a.set_fontweight("bold")
    # Direct labels of historical restrictions; positions are in screen points.
    callout(ax, points[0], "A", (-9, -3), INK,
            align="right", fontsize=12, leader=False)
    callout(ax, points[0], "May\n"+r"$\mu=\tau=0$", (-12, -28), INK,
            align="right", fontsize=11, leader=False)
    j_tau = 100.0
    callout(ax, (0, j_tau, 1), "Jirsa & Ding\n"+r"$\mu=0$", (0, 0),
            JIRSA, align="center", fontsize=11, figure_position=(0.20, 0.78))
    p_tau = 90.0
    callout(ax, (p_tau, p_tau, threshold(p_tau, p_tau)/R_0),
            "Pigani et al.\n"+r"$\mu=\tau$", (28, -10), PIGANI,
            fontsize=11)
    callout(ax, points[1], "B", (-19, -9), SELF, fontsize=12)
    callout(ax, points[2], "C", (8, 6), INTER, fontsize=12, leader=False)
    # Arrowheads follow each of the two independent-delay trajectories.
    for i0, i1, color, route in [(105, 128, SELF, "self"), (90, 112, INTER, "inter")]:
        if route == "self":
            start = (mu_path[i0], 0, self_r[i0])
            end = (mu_path[i1], 0, self_r[i1])
        else:
            start = (MU_FIXED, tau_path[i0], repair_r[i0])
            end = (MU_FIXED, tau_path[i1], repair_r[i1])
        arrow3d(ax, start, end, color)

    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"framework_landscape.{extension}", dpi=230,
                    bbox_inches="tight", pad_inches=0.12)
    print("Verified global-minimum surface and all three trajectory states.")
    for label, mu, tau in (("A", 0, 0), ("B", MU_FIXED, 0), ("C", MU_FIXED, TAU_FINAL)):
        R, omega = threshold(mu, tau, return_frequency=True)
        print(f"{label}: mu={mu:g}, tau={tau:g}, R={R:.9f}, omega_crit={omega:.9f}")


if __name__ == "__main__":
    main()





