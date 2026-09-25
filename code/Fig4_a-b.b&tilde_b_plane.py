import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
d_A = 1.0
mu = 1.326
r_B = 1.0
D = -0.8

# Five tau values
tau_values = [0.0, 0.5, 1.0, 1.5, 2.0]

# Five user-specified colors
colors = [
    (144 / 255, 201 / 255, 231 / 255),   # light blue
    (19 / 255, 103 / 255, 131 / 255),    # medium blue
    (0 / 255, 47 / 255, 73 / 255),       # dark blue
    (193 / 255, 18 / 255, 33 / 255),     # red
    (120 / 255, 0 / 255, 1 / 255),       # dark red
]

# ---------- Omega range set individually for each curve ----------
omega_ranges = [
    (-1.7, 1.7),  # tau = 0.0
    (-1.7, 1.7),  # tau = 0.5
    (-1.7, 1.7),  # tau = 1.0
    (-1.7, 1.7),  # tau = 1.5
    (-1.7, 1.7),  # tau = 2.0
]

# ---------- Helper functions ----------
def b_crit(omega, tau):
    """Critical curve in the b-plane."""
    return np.exp(1j * omega * tau) * (1j * omega + d_A - D * np.exp(-1j * omega * mu))

def b_tilde_crit(omega):
    """Critical curve in the tilde{b}-plane (independent of tau)."""
    return np.exp(1j * omega * mu) * (1j * omega + d_A)

# ============ First figure: b-plane ============
fig1, ax_left = plt.subplots(figsize=(6, 5))

theta = np.linspace(0, 2 * np.pi, 300)
ax_left.plot(r_B * np.cos(theta), r_B * np.sin(theta), 'k--', linewidth=2,
             label=r'eigenvalue disk $|b|\leq r_B$')
ax_left.plot(0, 0, 'ko', markersize=4)

all_curve_re = [0.0]
all_curve_im = [0.0]

for idx, (tau, col) in enumerate(zip(tau_values, colors)):
    om_min, om_max = omega_ranges[idx]
    omega = np.linspace(om_min, om_max, 2000)
    curve = b_crit(omega, tau)
    ax_left.plot(curve.real, curve.imag, color=col, linewidth=1.8,
                 label=r'$\tau=%.1f$' % tau)
    all_curve_re.extend(curve.real)
    all_curve_im.extend(curve.imag)

ax_left.set_title(r'$b$-plane', fontsize=12)
ax_left.set_xlabel(r'Re($b$)')
ax_left.set_ylabel(r'Im($b$)', labelpad=-6)
ax_left.set_aspect('equal')
ax_left.grid(alpha=0.3)

# Left panel coordinate limits
all_re_left = list(all_curve_re) + list(r_B * np.cos(theta))
all_im_left = list(all_curve_im) + list(r_B * np.sin(theta))
re_min_l, re_max_l = np.min(all_re_left), np.max(all_re_left)
im_min_l, im_max_l = np.min(all_im_left), np.max(all_im_left)
margin_l = 0.1 * max(re_max_l - re_min_l, im_max_l - im_min_l)
re_min_l -= margin_l
re_max_l += margin_l
im_min_l -= margin_l
im_max_l += margin_l
range_max_l = max(re_max_l - re_min_l, im_max_l - im_min_l)
re_center_l = (re_min_l + re_max_l) / 2
im_center_l = (im_min_l + im_max_l) / 2
ax_left.set_xlim(re_center_l - range_max_l / 2, re_center_l + range_max_l / 2)
ax_left.set_ylim(im_center_l - range_max_l / 2, im_center_l + range_max_l / 2)

plt.tight_layout()
plt.savefig('b_plane_single.svg', format='svg', bbox_inches='tight')
plt.close(fig1)

# ============ Second figure: \tilde{b}-plane ============
fig2, ax_right = plt.subplots(figsize=(6, 5))

omega_tilde = np.linspace(-1.7, 1.7, 2000)
curve_tilde = b_tilde_crit(omega_tilde)

center = D + 0j
ax_right.plot(center.real, center.imag, 'o', color='black', markersize=6)
ax_right.plot(center.real + r_B * np.cos(theta), center.imag + r_B * np.sin(theta),
              'k--', linewidth=2, label=r'eigenvalue disk $|\tilde{b}-D|\leq r_B$')
ax_right.plot(curve_tilde.real, curve_tilde.imag, 'k-', linewidth=2,
              label=r'$\tilde{\gamma}(\omega)$')

ax_right.set_title(r'$\tilde{b}$-plane', fontsize=12)
ax_right.set_xlabel(r'Re($\tilde{b}$)')
ax_right.set_ylabel(r'Im($\tilde{b}$)', labelpad=-6)
ax_right.set_aspect('equal')
ax_right.grid(alpha=0.3)

# Right panel coordinate limits
all_re_right = list(curve_tilde.real) + list(center.real + r_B * np.cos(theta)) + [center.real]
all_im_right = list(curve_tilde.imag) + list(center.imag + r_B * np.sin(theta)) + [center.imag]
re_min_r, re_max_r = np.min(all_re_right), np.max(all_re_right)
im_min_r, im_max_r = np.min(all_im_right), np.max(all_im_right)
margin_r = 0.1 * max(re_max_r - re_min_r, im_max_r - im_min_r)
re_min_r -= margin_r
re_max_r += margin_r
im_min_r -= margin_r
im_max_r += margin_r
range_max_r = max(re_max_r - re_min_r, im_max_r - im_min_r)
re_center_r = (re_min_r + re_max_r) / 2
im_center_r = (im_min_r + im_max_r) / 2
ax_right.set_xlim(re_center_r - range_max_r / 2, re_center_r + range_max_r / 2)
ax_right.set_ylim(im_center_r - range_max_r / 2, im_center_r + range_max_r / 2)

plt.tight_layout()
plt.savefig('btilde_plane_single.svg', format='svg', bbox_inches='tight')
plt.close(fig2)

plt.show()