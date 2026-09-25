import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

d_A = 0.2
D = -2.0
d_AD = d_A - D
omega_max = 10.0

def calc_R(alpha, k):
    mu = k * alpha
    tau = (1 - k) * alpha

    def dist_x(x):
        omega = x / alpha
        if omega < 0 or omega > omega_max:
            return np.inf
        b = (1 + 1j * omega * tau) * (1j * omega + d_A - D / (1 + 1j * omega * mu))
        return np.abs(b)

    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]

    res = minimize_scalar(
        dist_x,
        bounds=(max(0, x0 - 1), min(50, x0 + 1)),
        method='bounded'
    )
    min_val = res.fun if res.success else coarse_v[idx]
    val0 = dist_x(0.0)
    return min(min_val, val0)

def compute_R_star(k, alpha_vals):
    R_curve = [calc_R(alpha, k) for alpha in alpha_vals]
    return np.min(R_curve)

def compute_R_infty(k):
    if np.isclose(k, 1.0):
        return d_A
    C = (-D) * (2 * d_A + (-D))

    def f(x):
        F = (1 + (1 - k) ** 2 * x ** 2) * (d_A ** 2 + C / (1 + k ** 2 * x ** 2))
        return np.sqrt(F)

    x_max = 20.0
    while True:
        xs = np.linspace(0, x_max, 2000)
        fs = f(xs)
        idx = np.argmin(fs)
        x0 = xs[idx]
        if x0 > 0.95 * x_max:
            x_max *= 2.0
        else:
            break
    res = minimize_scalar(
        f,
        bounds=(max(0, x0 - max(1, x0 * 0.1)), x0 + max(1, x0 * 0.1)),
        method='bounded'
    )
    return res.fun if res.success else f(x0)

k_vals = np.linspace(0, 1, 1601)
rB_vals = np.linspace(0.01, 2.5, 3000)
alpha_vals = np.logspace(-1, 3.5, 300)

print("Computing R* and R_infty ...")
t0 = time.time()
R_star_arr = np.empty_like(k_vals)
R_infty_arr = np.empty_like(k_vals)
for i, k in enumerate(k_vals):
    if i % 40 == 0:
        print(f"  k = {k:.3f} ({i + 1}/{len(k_vals)})")
    R_star_arr[i] = compute_R_star(k, alpha_vals)
    R_infty_arr[i] = compute_R_infty(k)
print(f"Elapsed time {time.time() - t0:.1f}s")

cat = np.empty((len(rB_vals), len(k_vals)), dtype=np.int8)
for j, rB in enumerate(rB_vals):
    cat[j, :] = np.where(rB < R_star_arr, 0,
                         np.where(rB < R_infty_arr, 1,
                                  np.where(rB < d_AD, 2, 3)))

from matplotlib.colors import ListedColormap
gray_cmap = ListedColormap(['white', 'lightgrey', 'grey', 'dimgrey'])

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(cat, aspect='auto', origin='lower',
               extent=[0, 1, rB_vals[0], rB_vals[-1]],
               cmap=gray_cmap, vmin=-0.5, vmax=3.5,
               interpolation='nearest')

cbar = plt.colorbar(im, ticks=[0, 1, 2, 3], ax=ax, pad=0.02)
cbar.ax.set_yticklabels(['S', 'R', 'U1', 'U2'], fontsize=9)

for k_ref, label in zip([0.0, 0.5, 1.0],
                        ['Jirsa-Ding', 'Pigani', 'pure self-delay']):
    ax.axvline(x=k_ref, color='black', linestyle='--', linewidth=1.2)

ax.set_xlabel('k', fontsize=12)
ax.set_ylabel('$r_B$', fontsize=12)
ax.set_title('Phase diagram', fontsize=13)

plt.tight_layout(pad=0.5)
plt.savefig('phase_diagram.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()