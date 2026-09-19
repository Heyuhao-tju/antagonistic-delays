import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

matplotlib.use('TkAgg')  # or 'Agg' to save and exit

# ========== Fixed parameters ==========
d_A = 1.0
D = -0.8
C = 0.1
sigma = 0.1
S = 1000
r_B = sigma * np.sqrt(S * C)   # = 1.0

mu = 1.326

# Fixed tau values
tau_values = [0.0, 0.5, 1.0, 1.5, 2.0]

# Five user-specified colors (light to dark)
colors = [
    (144/255, 201/255, 231/255),   # light blue #90C9E7
    (19/255, 103/255, 131/255),    # medium blue #136783
    (0/255, 47/255, 73/255),       # dark blue #002F49
    (193/255, 18/255, 33/255),     # red #C11221
    (120/255, 0/255, 1/255),       # dark red #780001
]

# ========== Generate eigenvalues of random matrix B ==========
np.random.seed(42)
mask = np.random.rand(S, S) < C
np.fill_diagonal(mask, False)
B = np.zeros((S, S), dtype=complex)
B[mask] = np.random.normal(0, sigma, size=mask.sum()) + 0j
z_eig = np.linalg.eigvals(B)

print(f"r_B = {r_B:.4f}, actual max modulus = {np.max(np.abs(z_eig)):.4f}")

# ========== Solve characteristic equation lambda = -d_A + D e^{-lambda mu} + z e^{-lambda tau} ==========
def solve_lambda(z, mu, tau):
    """Solve lambda for given z, mu, tau; return NaN on failure."""
    if mu == 0 and tau == 0:
        return -d_A + D + z

    def equations(xy):
        L = xy[0] + 1j*xy[1]
        val = L + d_A - D * np.exp(-mu * L) - z * np.exp(-tau * L)
        return [val.real, val.imag]

    # Multiple initial guesses to improve convergence
    guesses = [
        [z.real - d_A, z.imag],
        [z.real - d_A + D, z.imag],
        [0.0, 0.0],
        [z.real, z.imag],
        [z.real - d_A, 0.0],
        [0.0, z.imag]
    ]
    for x0 in guesses:
        # Clip real part of initial guess to reduce overflow
        x0[0] = np.clip(x0[0], -5, 5)
        sol, info, ier, msg = fsolve(equations, x0, full_output=True, xtol=1e-8)
        if ier == 1:
            return sol[0] + 1j*sol[1]
    return np.nan + 1j*np.nan

def compute_lambdas(z_eig, mu, tau):
    lambdas = []
    for z in z_eig:
        lam = solve_lambda(z, mu, tau)
        if not np.isnan(lam.real):
            lambdas.append(lam)
    return np.array(lambdas)

# ========== Plot theoretical contour ==========
def plot_contour(ax, mu, tau, r_B, color):
    """Plot zero contour of |lambda + d_A - D e^{-lambda mu}|^2 * e^{2 Lambda tau} = r_B^2."""
    Lambda_vals = np.linspace(-3.5, 1.0, 1200)
    omega_vals = np.linspace(-2.2, 2.2, 1200)
    Lambda_grid, omega_grid = np.meshgrid(Lambda_vals, omega_vals)

    exp_mu = np.exp(-mu * Lambda_grid)
    term1 = Lambda_grid**2 + d_A**2 + omega_grid**2 + 2*Lambda_grid*d_A
    term2 = D**2 * np.exp(-2*mu*Lambda_grid)
    term3 = 2*D*exp_mu * (omega_grid*np.sin(omega_grid*mu) -
                           np.cos(omega_grid*mu)*(Lambda_grid + d_A))
    mod_sq = term1 + term2 + term3
    F = mod_sq * np.exp(2*tau*Lambda_grid) - r_B**2

    try:
        contour = ax.contour(Lambda_grid, omega_grid, F, levels=[0],
                             colors=[color], linewidths=2.5, alpha=0.9)
        # Mark rightmost point and its mirror about the real axis
        if hasattr(contour, 'allsegs') and contour.allsegs:
            all_segs = contour.allsegs[0]
            if all_segs:
                points = np.vstack(all_segs)
                max_idx = np.argmax(points[:, 0])
                ax.plot(points[max_idx, 0], points[max_idx, 1], 'o',
                        color=color, markersize=8, markeredgecolor='black')
                ax.plot(points[max_idx, 0], -points[max_idx, 1], 'o',
                        color=color, markersize=8, markeredgecolor='black')
    except Exception as e:
        print(f"Error plotting contour (mu={mu}, tau={tau}): {e}")

# ========== Plot on the same figure ==========
fig, ax = plt.subplots(figsize=(12, 10))

# Fill positive real part (Re(lambda) > 0) in gray to indicate unstable region
ax.axvspan(0, 1.0, color='gray', alpha=0.2, zorder=0)

for idx, tau in enumerate(tau_values):
    color = colors[idx]
    print(f"Processing tau = {tau} ...")

    lambdas = compute_lambdas(z_eig, mu, tau)
    print(f"  Successfully solved {len(lambdas)} eigenvalues")

    # Scatter points with low opacity to avoid hiding contours
    ax.scatter(lambdas.real, lambdas.imag, color=color, alpha=0.2, s=12,
               label=f'$\\tau = {tau}$')

    # Contour line with matching color
    plot_contour(ax, mu, tau, r_B, color)

# ========== Figure formatting ==========
ax.set_xlabel('Re(λ) = Λ', fontsize=16)
ax.set_ylabel('Im(λ) = ω', fontsize=16)

ax.grid(True, alpha=0.3, linestyle=':')
ax.set_xlim([-2.5, 1.0])
ax.set_ylim([-2.2, 2.2])
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('lambda_plane_multiple_tau.svg', format='svg', bbox_inches='tight')
plt.show()


