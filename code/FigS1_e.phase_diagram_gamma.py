import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import time

# Global font settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Parameters (Gamma kernel n=2)
d_A = 0.4
D = -1.8
d_AD = d_A - D               # May bound = 2.2
omega_max = 10.0

# Gamma kernel Laplace transform
def L_gamma(omega, delay):
    return (1 + 1j * omega * delay / 2.0) ** (-2)

# Compute R(alpha, k) using scaled variable x = omega*alpha
def calc_R(alpha, k):
    mu = k * alpha
    tau = (1 - k) * alpha

    def dist_x(x):
        omega = x / alpha
        if omega < 0 or omega > omega_max:
            return np.inf
        L_mu = L_gamma(omega, mu)
        L_tau = L_gamma(omega, tau)
        b = (1j * omega + d_A - D * L_mu) / L_tau
        return np.abs(b)

    # Coarse grid search in scaled variable x
    coarse_x = np.linspace(0, 20, 1000)
    coarse_v = [dist_x(x) for x in coarse_x]
    idx = np.argmin(coarse_v)
    x0 = coarse_x[idx]

    # Local refinement
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
    # Directly minimize the limiting function F_inf(x;k)
    def F_inf(x):
        # Limiting function for Gamma n=2
        A = (1 + 1j * (1 - k) * x / 2.0) ** 2
        B = d_A - D * (1 + 1j * k * x / 2.0) ** (-2)
        return abs(A * B)

    # Coarse scan
    xs = np.linspace(0, 20, 2000)
    ys = [F_inf(x) for x in xs]
    idx = np.argmin(ys)
    x0 = xs[idx]

    # Local refinement
    res = minimize_scalar(
        F_inf,
        bounds=(max(0, x0 - 1), min(50, x0 + 1)),
        method='bounded'
    )
    return res.fun if res.success else ys[idx]

# Grids
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

# Classification
cat = np.empty((len(rB_vals), len(k_vals)), dtype=np.int8)
for j, rB in enumerate(rB_vals):
    cat[j, :] = np.where(rB < R_star_arr, 0,
                         np.where(rB < R_infty_arr, 1,
                                  np.where(rB < d_AD, 2, 3)))

# Custom grayscale colormap
from matplotlib.colors import ListedColormap
gray_cmap = ListedColormap(['white', 'lightgrey', 'grey', 'dimgrey'])

# Plot (wide format)
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(cat, aspect='auto', origin='lower',
               extent=[0, 1, rB_vals[0], rB_vals[-1]],
               cmap=gray_cmap, vmin=-0.5, vmax=3.5,
               interpolation='nearest')

cbar = plt.colorbar(im, ticks=[0, 1, 2, 3], ax=ax, pad=0.02)
cbar.ax.set_yticklabels(['S', 'R', 'U1', 'U2'], fontsize=9)

ax.set_xlabel('k', fontsize=12)
ax.set_ylabel('$r_B$', fontsize=12)
ax.set_title('Phase diagram ($\\Gamma$-kernel $n=2$)', fontsize=13)

plt.tight_layout(pad=0.5)
plt.savefig('phase_diagram_gamma.svg', format='svg',
            bbox_inches='tight', pad_inches=0.05)
plt.show()