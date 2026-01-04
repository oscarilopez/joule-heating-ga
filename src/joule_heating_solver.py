import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Givens
# ---------------------------

# Domain
L = 0.01  # m
T_final = 1.0
dt = 1e-5
N = int(T_final / dt)
Ms = [2, 10, 20]
num_cases = 4  # Right-hand boundary condition cases: A, B, C, D
Letters = ['A', 'B', 'C', 'D']

# Material properties
C = 500       # specific heat capacity [J/K/kg]
K = 60        # thermal conductivity [W/m/K]
rho = 7850    # density [kg/m^3]
sigmac = 7e6  # electrical conductivity [S/m]

# Joule heating settings
theta0 = 300.0       # initial temperature [K]
a = 0.8              # efficiency
delta = -1e6         # non-zero boundary flux [W/m^2]
E = 15.0             # electric field [V/m]
J = sigmac * E       # current density [A/m^2]

# Output directory
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------
# Steady-State Solutions
# ---------------------------

# 2nd order polynomial coefficients: theta(x) = A*x^2 + B*x + C
As = (-(a * J * E) / (2 * K)) * np.ones(num_cases)
Bs = np.array([
    (((a * J * E) / (2 * K)) * L),
    (((a * J * E) / (2 * K)) * L) + 100 / L,
    (((a * J * E) / (1 * K)) * L),
    (((a * J * E) / (1 * K)) * L) + delta / K
])
Cs = theta0 * np.ones(num_cases)

xx = np.linspace(0, L, 1000)
ss_solns = np.zeros((num_cases, len(xx)))

for Case in range(num_cases):
    ss_solns[Case, :] = As[Case] * xx**2 + Bs[Case] * xx + Cs[Case]

# ---------------------------
# Transient Forward Euler
# ---------------------------

joule_heating = a * J * E  # constant source term

for M in Ms:
    x = np.linspace(0, L, M + 1)
    dx = L / M

    for Case in range(num_cases):
        Letter = Letters[Case]
        title_str = f"Boundary Condition {Letter}, {M} discretizations"
        filename = os.path.join(FIG_DIR, f"{Letter}{M}.png")

        plt.figure(figsize=(8, 6))
        plt.plot(xx, ss_solns[Case, :], linewidth=3, label="Steady-State")

        theta = theta0 * np.ones(M + 1)

        for n in range(1, N + 1):
            theta_new = theta.copy()

            # Interior update only (avoids periodic wrap from np.roll)
            diffusion_interior = K * (theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dx**2
            theta_new[1:-1] = theta[1:-1] + dt / (rho * C) * (diffusion_interior + joule_heating)

            # Left boundary (Dirichlet)
            theta_new[0] = theta0

            # Right-hand boundary cases
            if Case == 0:  # A: fixed temperature
                theta_new[-1] = theta0
            elif Case == 1:  # B: prescribed temperature
                theta_new[-1] = theta0 + 100
            elif Case == 2:  # C: zero heat flux (Neumann)
                theta_new[-1] = theta_new[-2]
            else:  # D: constant cooling flux (Neumann)
                theta_new[-1] = theta_new[-2] + (delta / K) * dx

            theta = theta_new

            if n in [round(N/5), round(2*N/5), round(3*N/5), round(4*N/5)]:
                plt.plot(x, theta, linewidth=2, label=f"$t = {n*dt:.2f}$ s")

        plt.plot(x, theta, linewidth=3, label="$t=T$")
        plt.xlabel(r"$x$ position along bar (m)", fontsize=14)
        plt.ylabel(r"Temperature (K)", fontsize=14)
        plt.title(title_str, fontsize=16)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

print("Simulation complete. Plots saved in ./figures/")
