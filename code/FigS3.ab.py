import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d_A = 1.0
mu = 1.0
r_B = 1.0

def b_tilde_crit(omega):
    return np.exp(1j * omega * mu) * (1j * omega + d_A)

def b_orig_crit(omega, D, tau):
    return np.exp(1j * omega * tau) * (1j * omega + d_A - D * np.exp(-1j * omega * mu))

omega_plot = np.linspace(-3, 3, 1000)
theta = np.linspace(0, 2*np.pi, 300)

D_list_plot = [0, 0.2]
D_list_range = [-1.159, -1.359]

tau_vals_global = [0.0, 0.5, 1.0]

all_re = []
all_im = []

for tau in tau_vals_global:
    for D in D_list_range:
        bc = b_orig_crit(omega_plot, D, tau)
        all_re.extend(bc.real)
        all_im.extend(bc.imag)

bt = b_tilde_crit(omega_plot)
all_re.extend(bt.real)
all_im.extend(bt.imag)

for D in D_list_range:
    all_re.append(D)
    all_im.append(0.0)
    all_re.extend(D + r_B * np.cos(theta))
    all_im.extend(r_B * np.sin(theta))

all_re.extend(r_B * np.cos(theta))
all_im.extend(r_B * np.sin(theta))
all_re.append(0.0)
all_im.append(0.0)

re_min, re_max = np.min(all_re), np.max(all_re)
im_min, im_max = np.min(all_im), np.max(all_im)
margin = 0.1 * max(re_max - re_min, im_max - im_min)
re_min -= margin
re_max += margin
im_min -= margin
im_max += margin
range_max = max(re_max - re_min, im_max - im_min)
re_center = (re_min + re_max) / 2
im_center = (im_min + im_max) / 2
xlim = [re_center - range_max/2, re_center + range_max/2]
ylim = [im_center - range_max/2, im_center + range_max/2]

colors = ['#0346a5', '#c72a2a']

gamma_curve_labels = [
    r'$\gamma(\omega)$, $D=0$',
    r'$\gamma(\omega)$, $D=0.2$'
]
b_disk_label = r'eigenvalue disk $|b|\leq r_B$'
tilde_gamma_label = r'$\tilde{\gamma}(\omega)$'
tilde_disk_labels = [
    r'eigenvalue disk $|\tilde{b}-D|\leq r_B$, $D=0$',
    r'eigenvalue disk $|\tilde{b}-D|\leq r_B$, $D=0.2$'
]

fig = plt.figure(figsize=(22, 5))

left_start = 0.05
bottom = 0.1
width = 0.18
height = 0.8
d1 = 0.03
d2 = 0.035

ax0 = fig.add_axes([left_start, bottom, width, height])
ax1 = fig.add_axes([left_start + width + d1, bottom, width, height])
ax2 = fig.add_axes([left_start + 2*(width + d1), bottom, width, height])
ax3 = fig.add_axes([left_start + 3*width + 2*d1 + d2, bottom, width, height])

axes_b = [ax0, ax1, ax2]
tau_vals = [0.0, 0.5, 1.0]

for ax, tau in zip(axes_b, tau_vals):
    ax.plot(r_B*np.cos(theta), r_B*np.sin(theta), 'k--', linewidth=2)
    ax.plot(0, 0, 'ko', markersize=4)
    for D, col in zip(D_list_plot, colors):
        curve = b_orig_crit(omega_plot, D, tau)
        ax.plot(curve.real, curve.imag, color=col, linewidth=1.8)
    ax.set_title(r'b-plane$(\tau = %.1f)$' % tau, fontsize=12)
    ax.set_xlabel(r'Re($b$)')
    ax.set_ylabel(r'Im($b$)', labelpad=-6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

legend_b = [
    Line2D([0], [0], color='k', linestyle='--', linewidth=2, label=b_disk_label),
    Line2D([0], [0], color=colors[0], linewidth=1.8, label=gamma_curve_labels[0]),
    Line2D([0], [0], color=colors[1], linewidth=1.8, label=gamma_curve_labels[1])
]
ax0.legend(handles=legend_b, loc='upper left', fontsize=5.6)

ax3.plot(bt.real, bt.imag, 'k-', linewidth=2)
for D, col in zip(D_list_plot, colors):
    center = D + 0j
    ax3.plot(center.real, center.imag, 'o', color=col, markersize=6, markerfacecolor=col)
    ax3.plot(center.real + r_B*np.cos(theta), center.imag + r_B*np.sin(theta),
             '--', color=col, linewidth=1.8)
ax3.set_title(r'$\tilde{b}$-plane', fontsize=12)
ax3.set_xlabel(r'Re($\tilde{b}$)')
ax3.set_ylabel(r'Im($\tilde{b}$)', labelpad=-6)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_aspect('equal')
ax3.grid(alpha=0.3)

legend_tilde = [
    Line2D([0], [0], color='k', linestyle='-', linewidth=2, label=tilde_gamma_label),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=8, label=tilde_disk_labels[0]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=8, label=tilde_disk_labels[1])
]
ax3.legend(handles=legend_tilde, loc='upper right', fontsize=5.6)

plt.savefig('combined_four_panels_2.svg', format='svg', bbox_inches='tight')
plt.show()
print(xlim, ylim)