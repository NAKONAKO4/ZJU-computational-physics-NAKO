import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
N = 12  # Number of oscillators
K = 1.0  # Coupling constant
m = 1.0  # Mass of oscillators
dt_values = [0.01, 0.05, 0.1]  # Time step sizes
T = 10  # Total simulation time

# Initial conditions
u = np.zeros(N)  # Initial positions
v = np.zeros(N)  # Initial velocities
v[2] = 1.0  # v3(t=0) = 1

# Define coupling matrix
def coupling_matrix(N, K):
    M = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            M[i, i - 1] = K
        M[i, i] = -2 * K
        if i < N - 1:
            M[i, i + 1] = K
    return M

M = coupling_matrix(N, K)

# Define total energy calculation
def total_energy(u, v, K):
    kinetic = 0.5 * m * np.sum(v**2)
    potential = 0.5 * K * np.sum((np.diff(u)**2))
    return kinetic + potential

# Numerical solution using Runge-Kutta 4th order method
def runge_kutta(u, v, M, dt):
    a = np.dot(M, u)
    k1_u = v
    k1_v = a / m

    k2_u = v + 0.5 * dt * k1_v
    k2_v = np.dot(M, u + 0.5 * dt * k1_u) / m

    k3_u = v + 0.5 * dt * k2_v
    k3_v = np.dot(M, u + 0.5 * dt * k2_u) / m

    k4_u = v + dt * k3_v
    k4_v = np.dot(M, u + dt * k3_u) / m

    u_next = u + (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
    v_next = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return u_next, v_next

# Simulate and analyze
for dt in dt_values:
    u = np.zeros(N)
    v = np.zeros(N)
    v[2] = 1.0

    time = np.arange(0, T, dt)
    energies = []
    max_deviation = 0

    for t in time:
        # Analytical solution (for comparison, simplified for example)
        u_analytic = np.zeros(N)  # Replace with actual analytic solution if available

        # Runge-Kutta step
        u, v = runge_kutta(u, v, M, dt)

        # Calculate energy and maximum deviation
        energies.append(total_energy(u, v, K))
        deviation = np.max(np.abs(u - u_analytic))
        max_deviation = max(max_deviation, deviation)

    # Plot energy conservation
    plt.plot(time, energies, label=f"dt = {dt}")


plt.title("Energy Conservation in Coupled Oscillators")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.legend()
plt.grid()
plt.show()