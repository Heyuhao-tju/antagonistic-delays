import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, bisect

d_A = 1.0
d_B = 0.8
r_B_list = [0.15, 1.0, 2.0]
tau_list = [0.0, 0.5, 1.0, 2.0, 5.0]

stable_color = '#0346a5'
unstable_color = '#c72a2a'

def eqs(vars, r_B_val, d_A_val, d_B_val):
    tau, omega = vars
    eq1 = (d_A_val**2 + d_B_val**2 + omega**2 +
           2*d_B_val*(d_A_val*np.cos(omega*tau) - omega*np.sin(omega*tau)) - r_B_val**2)
    eq2 = ((1 + d_A_val*tau)*d_B_val*np.sin(omega*tau) +
           d_B_val*omega*tau*np.cos(omega*tau) - omega)
    return [eq1, eq2]

r_B_mid = 1.0
sol = fsolve(lambda x: eqs(x, r_B_mid, d_A, d_B), [1.2, 0.8])
tau_crit = sol[0]
print(f"Critical tau for r_B=1: {tau_crit:.4f}")

def is_stable(r_B_val, tau_val):
    if r_B_val <= max(0, d_A - d_B):
        return True
    if r_B_val >= d_A + d_B:
        return False
    # Solve critical tau for the given r_B_val
    sol_local = fsolve(lambda x: eqs(x, r_B_val, d_A, d_B), [1.0, 0.5])
    tau_crit_local = sol_local[0]
    return tau_val < tau_crit_local

def find_first_omega_max(tau, d_A, bracket_start=0.1, bracket_end=None):
    """Return the smallest omega > 0 such that Im(b(omega)) = 0."""
    if bracket_end is None:
        bracket_end = 10.0
    def im_b(omega):
        b = np.exp(1j * omega * tau) * (1j * omega + d_A)
        return np.imag(b)
    omega_scan = np.linspace(0.001, bracket_end, 500)
    im_vals = im_b(omega_scan)
    sign_changes = np.where(np.diff(np.sign(im_vals)))[0]
    if len(sign_changes) == 0:
        return bracket_end
    else:
        idx = sign_changes[0]
        a = omega_scan[idx]
        b = omega_scan[idx+1]
        omega_root = bisect(im_b, a, b, xtol=1e-6)
        return omega_root

tau_positions = {
    0.0: (0.7, 0.5),   # upper left
    0.5: (0.05, 0.2),   # upper right
    1.0: (0.08, 0.4),   # lower left
    2.0: (0.25, 0.4),   # lower right
    5.0: (0.38, 0.38),   # top center
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, r_B_val in enumerate(r_B_list):
    ax = axes[idx]
    ax.set_title(f"$r_B = {r_B_val}$", fontsize=14)
    ax.set_xlabel("Re(b)")
    ax.set_ylabel("Im(b)")
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Draw the always-stable circle (Jirsa–Ding)
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(d_A * np.cos(theta), d_A * np.sin(theta),
            '--', color='black', linewidth=1.2)

    # Draw the eigenvalue disk of matrix B (centered at -d_B, radius r_B)
    disk = plt.Circle((-d_B, 0), r_B_val, fill=True,
                      facecolor='lightgray', edgecolor='black',
                      linewidth=1.0, alpha=0.5)
    ax.add_patch(disk)

    for tau_val in tau_list:
        # Find omega_max to get the teardrop branch
        omega_max = find_first_omega_max(tau_val, d_A)
        omega_vals = np.linspace(0, omega_max, 1000)
        b_vals = np.exp(1j * omega_vals * tau_val) * (1j * omega_vals + d_A)
        # Add complex conjugate to form closed curve
        b_full = np.concatenate([b_vals, np.conj(b_vals[-2::-1])])

        stable = is_stable(r_B_val, tau_val)
        color = stable_color if stable else unstable_color

        ax.plot(np.real(b_full), np.imag(b_full), color=color, linewidth=1.2)

        # Place tau annotation at a corner/edge using axes fraction
        if tau_val in tau_positions:
            x_frac, y_frac = tau_positions[tau_val]
            ax.annotate(f"$\\tau={tau_val}$",
                        xy=(x_frac, y_frac), xycoords='axes fraction',
                        ha='left', va='top', color=color, fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.8))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('teardrop_rB.svg', format='svg', bbox_inches='tight')
plt.show()