import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Parameters
d_A = 1.0   # fixed by rescaling
omega_max = 50.0

def b_abs(omega, alpha, k, D):
    if omega < 0:
        return np.inf
    mu = k * alpha
    tau = (1 - k) * alpha
    b = (1 + 1j * omega * tau) * (1j * omega + d_A - D / (1 + 1j * omega * mu))
    return np.abs(b)

def find_min_omega(alpha, k, D):
    res = minimize_scalar(lambda w: b_abs(w, alpha, k, D),
                          bounds=(0, omega_max),
                          method='bounded',
                          options={'xatol': 1e-12, 'maxiter': 1000})
    if res.success:
        w_opt = res.x
        val_opt = res.fun
    else:
        omegas = np.linspace(0, omega_max, 2000)
        vals = np.array([b_abs(w, alpha, k, D) for w in omegas])
        idx = np.argmin(vals)
        w_opt = omegas[idx]
        val_opt = vals[idx]

    val0 = b_abs(0, alpha, k, D)
    if val0 <= val_opt:
        return 0.0, val0, False
    else:
        return w_opt, val_opt, True

# Scan parameters
k_values = np.linspace(0, 1, 101)
alpha_values = np.logspace(-1, 3, 500)     # 0.1 to 1000

# q values: logarithmic from 0.01 to 100 in magnitude, negative
abs_q_values = np.logspace(-2, 2, 100)  # 0.01 to 100
q_values = -abs_q_values

min_B_list = []

for q in q_values:
    D = q
    all_B = []
    for k in k_values:
        R_alpha = np.zeros_like(alpha_values)
        omega_opt = np.zeros_like(alpha_values)
        interior_flags = np.zeros_like(alpha_values, dtype=bool)

        for i, alpha in enumerate(alpha_values):
            w_opt, val_opt, interior = find_min_omega(alpha, k, D)
            R_alpha[i] = val_opt
            omega_opt[i] = w_opt
            interior_flags[i] = interior

        # global minimum over positive alpha
        idx_star = np.argmin(R_alpha)
        R_star_positive = R_alpha[idx_star]

        # If the minimum is essentially the May bound, then no interior minimizer
        R0 = d_A - D   # = 1 - D
        if abs(R_star_positive - R0) < 1e-6:
            continue   # skip this k, no interior minimizer

        # Otherwise, collect B at this interior minimizer
        w_star = omega_opt[idx_star]
        alpha_star = alpha_values[idx_star]
        mu_star = k * alpha_star
        omega_mu = w_star * mu_star
        B_val = 1 + (2*d_A - D) * mu_star - omega_mu**2
        all_B.append(B_val)

    if all_B:
        min_B = min(all_B)
        min_B_list.append(min_B)
    else:
        min_B_list.append(np.nan)   # no interior minimizer for this q

    print(f"q = {q:8.4f}, min_B = {min_B if all_B else 'N/A'}")

# Plot
plt.figure(figsize=(8, 5))
plt.semilogx(np.abs(q_values), min_B_list, 'o-', linewidth=2, markersize=4)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='$B=0$')
plt.xlabel(r'$|q| = |D|/d_A$', fontsize=14)
plt.ylabel(r'$\min_k B$', fontsize=14)
plt.title(r'Minimum $B$ over all interior minimizers for different $q$')
plt.grid(alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('minB_vs_q.svg', format='svg', bbox_inches='tight')
plt.show()