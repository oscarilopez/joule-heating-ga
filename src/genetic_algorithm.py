import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ==============================================================
# SIMULATION FUNCTION: joule_heating()
# ==============================================================

def joule_heating(Lambda):
    """
    Simulates transient 1D Joule heating for given material and process parameters
    and returns a scalar cost.
    """
    # Design variables
    K, delta, E, rho, sigmac, C = Lambda

    # Cost function targets
    theta_des1 = 569  # desired max peak temp [K]
    theta_des2 = 559  # desired edge temp [K]
    w1, w2 = 1000, 1000  # weights

    # Domain and discretization
    L = 0.01
    T = 1.0
    dt = 1e-5
    N = int(T / dt)
    M = 20

    # Joule heating parameters
    theta0 = 300.0
    a = 0.8

    # Source term a * sigma_c * E^2 (constant)
    q = a * sigmac * E**2

    # Discretize spatial domain
    dx = float(L / M)

    # Initial condition
    theta = theta0 * np.ones(M + 1)

    # Forward Euler time-stepping (interior update only)
    for _ in range(N):
        theta_new = theta.copy()

        diffusion_interior = K * (theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dx**2
        theta_new[1:-1] = theta[1:-1] + dt / (rho * C) * (diffusion_interior + q)

        # Boundary conditions (Case D: constant cooling flux)
        theta_new[0] = theta0
        theta_new[-1] = float(theta_new[-2] + (delta / K) * dx)

        theta = theta_new

    # Extract quantities for cost evaluation
    max_theta = np.max(theta)
    theta_L = theta[-1]

    # Cost function
    cost = (
        w1 * (abs(max_theta - theta_des1) / theta_des1) ** 3
        + w2 * (abs(theta_L - theta_des2) / theta_des2) ** 3
    )
    return cost

# ==============================================================
# GENETIC ALGORITHM OPTIMIZER
# ==============================================================

def optimize_joule_heating():
    start_time = datetime.now()

    # GA parameters
    P = 10          # number of parents (must be even)
    TOL = 1e-7
    G = 20
    S = 30
    dv = 6

    # Design variable bounds
    bounds = np.array([
        [100, 350],                 # K
        [-5e6, -1e6],               # delta
        [1, 15],                    # E
        [1000, 8000],               # rho
        [1e6, 50e6],                # sigmac
        [500, 1200],                # C
    ])

    scale_factor = bounds[:, 1] - bounds[:, 0]
    offset = bounds[:, 0]

    # Initialize population (dv x S)
    Lambda = np.random.rand(dv, S) * scale_factor[:, None] + offset[:, None]

    # Evaluate initial generation
    cost = np.zeros(S)
    for i in range(S):
        cost[i] = joule_heating(Lambda[:, i])

    ind = np.argsort(cost)
    cost = cost[ind]
    Lambda = Lambda[:, ind]

    # Tracking
    PI_best = np.full(G, np.inf)
    PI_avg = np.full(G, np.inf)
    PI_par_avg = np.full(G, np.inf)

    PI_best[0] = np.min(cost)
    PI_avg[0] = np.mean(cost)
    MIN = np.min(cost)

    top_performers = Lambda[:, :4].copy()
    top_costs = cost[:4]

    g = 1
    while MIN > TOL and g < G:
        g += 1
        print(f"Generation {g}")

        if P % 2 != 0:
            raise ValueError("P must be even for pairing.")

        parents = Lambda[:, :P]
        kids = np.zeros((dv, P))

        for p in range(0, P, 2):
            phi1, phi2 = np.random.rand(2)
            kids[:, p] = phi1 * parents[:, p] + (1 - phi1) * parents[:, p + 1]
            kids[:, p + 1] = phi2 * parents[:, p] + (1 - phi2) * parents[:, p + 1]

        # Random new designs
        new_strings = np.random.rand(dv, S - 2 * P) * scale_factor[:, None] + offset[:, None]

        # Parent fitness (for tracking only)
        par_cost = np.zeros(P)
        for i in range(P):
            par_cost[i] = joule_heating(parents[:, i])

        # New generation
        Lambda = np.concatenate((parents, kids, new_strings), axis=1)

        cost = np.zeros(S)
        for i in range(S):
            cost[i] = joule_heating(Lambda[:, i])

        ind = np.argsort(cost)
        cost = cost[ind]
        Lambda = Lambda[:, ind]

        PI_best[g - 1] = np.min(cost)
        PI_avg[g - 1] = np.mean(cost)
        PI_par_avg[g - 1] = np.mean(par_cost)

        MIN = min(MIN, np.min(cost))

        top_performers = Lambda[:, :4].copy()
        top_costs = cost[:4]

        print(f"Best cost this gen: {PI_best[g - 1]:.3e}")
        print("Top performer variables:", Lambda[:, 0])

    print("Runtime:", datetime.now() - start_time)

    # Plot convergence (save instead of show)
    plt.figure(figsize=(8, 6))
    gens = np.arange(1, g + 1)
    plt.semilogy(gens, PI_best[:g], linewidth=2, label="Best")
    plt.semilogy(gens, PI_avg[:g], linewidth=2, label="Overall Mean")
    plt.semilogy(gens, PI_par_avg[:g], linewidth=2, label="Parent Mean")
    plt.xlabel("Generations")
    plt.ylabel("Cost")
    plt.title("Joule Heating of a Bar: Convergence of Cost Function")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ga_convergence.png"), dpi=300)
    plt.close()

    print("\nTop Performing Designs (S1–S4):")
    for i in range(4):
        print(f"S{i+1}: {top_performers[:, i]}")
    print("\nCosts:", top_costs)

    return top_performers[:, 0]

# ==============================================================
# MAIN EXECUTION
# ==============================================================

def main():
    best_design = optimize_joule_heating()
    print("Optimization complete. Best design found:")
    print(best_design)

if __name__ == "__main__":
    main()
