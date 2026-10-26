import numpy as np


S = 1000
SIGMA = 0.1
C = 0.1
D_A = 0.2
D = -2.0
R_B = SIGMA * np.sqrt(S * C)
RHOS = (-0.75, -0.50, 0.0, 0.50, 0.75)
STAGES = ((0.0, 0.0), (50.0, 0.0), (50.0, 50.0))
N_THETA = 721


def characteristic_roots(b, mu, tau):
    """Solve lambda + d_A - D/(1+mu*lambda) - b/(1+tau*lambda) = 0.

    This script implements the three delay stages used in the table.
    For mu=tau>0, cancel the common denominator before solving.
    """
    b = np.asarray(b, dtype=complex)
    constant = D_A - D - b
    if mu == 0 and tau == 0:
        return (-constant)[:, None]
    if mu > 0 and tau == 0:
        # mu*lambda^2 + [1+mu*(d_A-b)]*lambda + d_A-D-b = 0.
        A = mu
        B = 1.0 + mu * (D_A - b)
    elif mu > 0 and tau == mu:
        # mu*lambda^2 + (1+mu*d_A)*lambda + d_A-D-b = 0.
        A = mu
        B = np.full(b.shape, 1.0 + mu * D_A, dtype=complex)
    else:
        raise ValueError("Supported stages: (0,0), (mu,0), (mu,mu).")
    discriminant = np.sqrt(B**2 - 4.0 * A * constant + 0j)
    return np.column_stack((
        (-B + discriminant) / (2.0 * A),
        (-B - discriminant) / (2.0 * A),
    ))


def boundary_max_real(rho, mu, tau):
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA)
    b = R_B * (
        (1.0 + rho) * np.cos(theta)
        + 1j * (1.0 - rho) * np.sin(theta)
    )
    roots = characteristic_roots(b, mu, tau)
    # Check the roots in the original rational equation.
    self_term = -D / (1.0 + mu * roots)
    inter_term = b[:, None] / (1.0 + tau * roots)
    residual = roots + D_A + self_term - inter_term
    scale = 1.0 + np.abs(roots) + D_A + np.abs(self_term) + np.abs(inter_term)
    assert np.max(np.abs(residual) / scale) < 1e-10
    rightmost = roots[np.arange(b.size), np.argmax(roots.real, axis=1)]
    return float(rightmost.real.max())


def main():
    print(f"Ellipse-boundary sampling: {N_THETA} angles; r_B={R_B:g}")
    print("Stages (mu, tau):", STAGES)
    print("\nSix decimal places:")
    print("rho       no delay     self delay      repair")
    results = []
    for rho in RHOS:
        values = [boundary_max_real(rho, mu, tau) for mu, tau in STAGES]
        results.append((rho, values))
        print(f"{rho:+.2f}   " + "   ".join(f"{v:+.6f}" for v in values))
    print("\nOriginal table, rounded to three decimal places:")
    print("| rho | No delay | Self-delay | Interaction-delay repair |")
    print("| --- | --- | --- | --- |")
    for rho, values in results:
        print(f"| {rho:g} | " + " | ".join(f"{v:+.3f}" for v in values) + " |")


if __name__ == "__main__":
    main()
