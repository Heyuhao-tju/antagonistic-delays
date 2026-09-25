import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Parameters
d_A = 1.0
D = -0.8
r_B = 1.0

alpha_min, alpha_max = 0.1, 100
num_alpha = 1000
num_theta = 100

k_vals = np.arange(0, 1.01, 0.1)

max_iter = 80
tol = 1e-12


def solve_lambda_robust(mu, tau, theta, prev_lambda=None):
    """
    Solve lambda + d_A - D exp(-lambda mu) - r_B exp(i theta) exp(-lambda tau) = 0
    """
    K2 = r_B * np.exp(1j * theta)

    def f(lam):
        exp_mu = np.exp(-lam * mu) if mu > 0 else 1.0
        exp_tau = np.exp(-lam * tau) if tau > 0 else 1.0
        return lam + d_A - D * exp_mu - K2 * exp_tau

    def f_prime(lam):
        exp_mu = np.exp(-lam * mu) if mu > 0 else 1.0
        exp_tau = np.exp(-lam * tau) if tau > 0 else 1.0
        return 1.0 + mu * D * exp_mu + tau * K2 * exp_tau

    init_guesses = []
    if prev_lambda is not None:
        init_guesses.append(prev_lambda)

    init_guesses.append(-d_A + D + K2)
    init_guesses.append(-d_A + 0j)
    init_guesses.append(-1.0 + 0j)

    if mu < 1e-3 and tau < 1e-3:
        init_guesses.append(-d_A + D + K2 / (1 + tau * K2))
    if mu < 1e-6:
        init_guesses.append(-d_A + D + K2)
    if tau < 1e-6:
        init_guesses.append(-d_A + K2 + D)

    best_lam = None
    best_residual = np.inf

    for lam0 in init_guesses:
        lam = lam0
        converged = False
        for _ in range(max_iter):
            try:
                fv = f(lam)
                fpv = f_prime(lam)
                if np.abs(fpv) < 1e-14:
                    break
                delta = fv / fpv
                lam -= delta
                if np.abs(delta) < tol:
                    converged = True
                    break
            except:
                break
        if converged:
            residual = np.abs(f(lam))
            if residual < best_residual:
                best_residual = residual
                best_lam = lam
            if residual < 1e-8:
                break

    if best_lam is None:
        return None

    if best_lam.real > 10 or best_lam.real < -10:
        return None
    return best_lam


alpha_vals = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_alpha)

all_curves = []
critical_mu = {}

for k in k_vals:
    print(f"k = {k:.1f}")
    max_real_vals = []
    bad_count = 0
    critical_alpha = None
    prev_best_lambda = None

    for idx, alpha in enumerate(alpha_vals):
        mu = k * alpha
        tau = (1 - k) * alpha
        max_real = -np.inf
        best_lambda_this_alpha = None

        for j in range(num_theta):
            theta = 2.0 * np.pi * j / num_theta
            lam = solve_lambda_robust(mu, tau, theta, prev_best_lambda)
            if lam is None:
                continue
            real_part = lam.real
            if -10 <= real_part <= 10:
                if real_part > max_real:
                    max_real = real_part
                    best_lambda_this_alpha = lam
            else:
                bad_count += 1

        if max_real == -np.inf:
            if len(max_real_vals) > 0:
                max_real = max_real_vals[-1]
                best_lambda_this_alpha = prev_best_lambda
            else:
                max_real = -np.inf
        else:
            prev_best_lambda = best_lambda_this_alpha

        max_real_vals.append(max_real)

        if critical_alpha is None and len(max_real_vals) >= 2:
            if max_real_vals[-2] < 0 and max_real >= 0:
                alpha_prev = alpha_vals[idx - 1]
                real_prev = max_real_vals[-2]
                if max_real - real_prev != 0:
                    critical_alpha = alpha_prev + (0 - real_prev) * (alpha - alpha_prev) / (max_real - real_prev)
                else:
                    critical_alpha = alpha
            if idx == 0 and max_real >= 0:
                critical_alpha = alpha

        if (idx + 1) % 100 == 0:
            print(f"  alpha = {alpha:.2f}, max real = {max_real:.6f}")

    all_curves.append((k, np.array(max_real_vals)))
    print(f"k = {k:.1f} finished, bad points: {bad_count}")
    if critical_alpha is not None:
        mu_star = k * critical_alpha
        print(f"  critical mu* = {mu_star:.6f} (alpha* = {critical_alpha:.6f})")
        critical_mu[k] = mu_star
    else:
        if max_real_vals[-1] < 0:
            print("  always stable")
        else:
            print("  always unstable")
        critical_mu[k] = None


plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(k_vals)))

for idx, (k, y_vals) in enumerate(all_curves):
    mu_star = critical_mu.get(k)
    if mu_star is not None:
        label = rf'$k = {k:.1f}$, $\mu^* = {mu_star:.3f}$'
    else:
        if y_vals[-1] < 0:
            label = rf'$k = {k:.1f}$ (always stable)'
        else:
            label = rf'$k = {k:.1f}$ (always unstable)'
    plt.plot(alpha_vals, y_vals, color=colors[idx], linewidth=1.5, label=label)

plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
plt.xlabel(r'$\tau + \mu$', fontsize=14)
plt.ylabel(r'$\Lambda_{\max}$', fontsize=14)
plt.xscale('log')
plt.xlim(alpha_min, alpha_max)
plt.ylim(-2, 0.5)

plt.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig('ratio_k.svg', format='svg', bbox_inches='tight')
plt.show()