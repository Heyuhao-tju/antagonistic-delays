import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Parameters
d_A = 0.2
D = -2
omega_max = 20.0
R0 = d_A - D   # 2.2

def b_abs(omega, alpha, k):
    if omega < 0:
        return np.inf
    mu = k * alpha
    tau = (1 - k) * alpha
    b = (1 + 1j * omega * tau) * (1j * omega + d_A - D / (1 + 1j * omega * mu))
    return np.abs(b)

def find_min_omega(alpha, k):
    """Return the omega that minimizes |b| for fixed alpha,k.
    Also return the minimum value and whether it is an interior minimizer."""
    # Coarse grid
    omegas = np.linspace(0, omega_max, 2000)
    vals = np.array([b_abs(w, alpha, k) for w in omegas])
    idx_min = np.argmin(vals)
    w0 = omegas[idx_min]

    # Local refinement
    res = minimize_scalar(lambda w: b_abs(w, alpha, k),
                          bounds=(max(0, w0 - 1), min(omega_max, w0 + 1)),
                          method='bounded')
    if res.success and res.fun < vals[idx_min]:
        w_opt = res.x
        val_opt = res.fun
    else:
        w_opt = w0
        val_opt = vals[idx_min]

    # Check if the optimum is significantly away from zero
    val0 = b_abs(0, alpha, k)
    if val0 <= val_opt:
        return 0.0, val0, False
    else:
        return w_opt, val_opt, True

# Scan parameters
k_values = np.linspace(0, 1, 101)
alpha_values = np.logspace(-1, 3, 500)     # 0.1 to 1000

omega_mu_at_star = []
alpha_star_list = []
omega_star_list = []
R_star_list = []
B_at_star = []

for k in k_values:
    R_alpha = np.zeros_like(alpha_values)
    omega_opt = np.zeros_like(alpha_values)
    interior_flags = np.zeros_like(alpha_values, dtype=bool)

    for i, alpha in enumerate(alpha_values):
        w_opt, val_opt, interior = find_min_omega(alpha, k)
        R_alpha[i] = val_opt
        omega_opt[i] = w_opt
        interior_flags[i] = interior

    # Find global minimum over positive alpha values
    idx_star = np.argmin(R_alpha)
    R_star_positive = R_alpha[idx_star]

    # If the minimum is essentially the May bound (within numerical error),
    # declare the minimizer as alpha = 0.
    if abs(R_star_positive - R0) < 1e-6:
        alpha_star = 0.0
        w_star = 0.0
        R_star = R0
        interior_star = False
    else:
        alpha_star = alpha_values[idx_star]
        w_star = omega_opt[idx_star]
        R_star = R_star_positive
        interior_star = interior_flags[idx_star]

        # Check if alpha* is at the scanned upper boundary
        if alpha_star >= alpha_values[-1] * 0.99:
            print(f"Warning: alpha* may be at upper boundary for k={k:.2f}, alpha*={alpha_star:.4f}")

    omega_mu = w_star * k * alpha_star
    omega_mu_at_star.append(omega_mu)
    alpha_star_list.append(alpha_star)
    omega_star_list.append(w_star)
    R_star_list.append(R_star)

    if interior_star:
        mu_star = k * alpha_star
        B_val = 1 + (2 * d_A - D) * mu_star - omega_mu**2
        B_at_star.append(B_val)
    else:
        B_at_star.append(np.nan)

    print(f"k={k:.2f}, alpha*={alpha_star:.4f}, omega*={w_star:.4f}, "
          f"omega*mu={omega_mu:.4f}, R*={R_star:.4f}, interior={interior_star}, "
          f"B={B_at_star[-1]:.6f}")

# Summarize B values
valid_B = [b for b in B_at_star if not np.isnan(b)]
if valid_B:
    min_B = min(valid_B)
    print(f"\nMinimum B across interior minimizers: {min_B:.6f}")
else:
    min_B = None
    print("\nNo interior minimizers found; B not defined.")

# Plot omega*mu
plt.figure(figsize=(8, 5))
plt.plot(k_values, omega_mu_at_star, 'o-', linewidth=2, markersize=4)
plt.xlabel(r'$k$', fontsize=14)
plt.ylabel(r'$\omega^* \mu$ at $\alpha^*$', fontsize=14)
plt.title(r'Value of $\omega\mu$ at the minimizer of $R(\alpha,k)$')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('omega_mu_vs_k.svg', dpi=300)
plt.show()

# Plot B
plt.figure(figsize=(8, 5))
plt.plot(k_values, B_at_star, 'o-', linewidth=2, markersize=4)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='$B=0$')
plt.xlabel(r'$k$', fontsize=14)
plt.ylabel(r'$B$ at $\alpha^*$', fontsize=14)
plt.title(r'Value of $B$ at the minimizer of $R(\alpha,k)$')
plt.grid(alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('B_vs_k.svg', dpi=300)
plt.show()

max_omega_mu = max(omega_mu_at_star)
print(f"\nMaximum omega*mu across all k: {max_omega_mu:.6f}")