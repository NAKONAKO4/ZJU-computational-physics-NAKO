import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
N = 12  # Number of oscillators
K = 1.0  # Coupling constant
m = 1.0  # Mass of oscillators
dt = 0.01  # Time step size
T = 20  # Total simulation time

# Initial conditions
u_initial = np.zeros(N)  # Initial positions
v_initial = np.zeros(N)  # Initial velocities
v_initial[2] = 1.0  # v3(t=0) = 1


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
    kinetic = 0.5 * m * np.sum(v ** 2)
    potential = 0.5 * K * np.sum((np.diff(u) ** 2))
    return kinetic + potential


# Analytical solution
def analytical_solution(N, K, m, time):
    # Frequencies and modes
    frequencies = np.array([2 * np.sqrt(K / m) * np.sin((k * np.pi) / (2 * (N + 1))) for k in range(1, N + 1)])
    modes = np.array([[np.sin((k * j * np.pi) / (N + 1)) for j in range(1, N + 1)] for k in range(1, N + 1)])

    # Determine amplitudes A_k using initial velocity
    A_k = np.array([
        (2 / (N + 1)) * np.sum(
            [np.sin((k * j * np.pi) / (N + 1)) * (1 if j == 3 else 0) for j in range(1, N + 1)]
        ) / frequencies[k - 1]
        for k in range(1, N + 1)
    ])

    # Solution at each time step
    u = np.zeros((len(time), N))
    for t_idx, t in enumerate(time):
        for j in range(N):
            u[t_idx, j] = np.sum(
                A_k * np.sin(((j + 1) * np.arange(1, N + 1) * np.pi) / (N + 1)) * np.cos(frequencies * t)
            )
    return u


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


# Numerical solution simulation
def runge_kutta_solution(N, K, m, dt, T, initial_u, initial_v, M):
    u = initial_u.copy()
    v = initial_v.copy()
    time = np.arange(0, T, dt)
    u_numerical = np.zeros((len(time), N))
    energies_numerical = []

    for t_idx, t in enumerate(time):
        u_numerical[t_idx] = u
        energies_numerical.append(total_energy(u, v, K))
        u, v = runge_kutta(u, v, M, dt)

    return time, u_numerical, energies_numerical


# Simulate numerical solution
time = np.arange(0, T, dt)
time, u_numerical, energies_numerical = runge_kutta_solution(N, K, m, dt, T, u_initial, v_initial, M)


# Analytical energy conservation
def analytical_energy(u_analytic, time, K, m):
    num_steps = len(time)
    energies_analytic = []

    for t_idx in range(num_steps):
        u = u_analytic[t_idx]
        if t_idx > 0:
            v = (u_analytic[t_idx] - u_analytic[t_idx - 1]) / dt
        else:
            v = np.zeros_like(u)  # Initial velocity is 0
        energy = total_energy(u, v, K)
        energies_analytic.append(energy)

    return energies_analytic


# Analytical solution and energy conservation
u_analytic = analytical_solution(N, K, m, time)
energies_analytic = analytical_energy(u_analytic, time, K, m)

# Plot energy conservation for both numerical and analytical solutions
plt.figure(figsize=(12, 6))
plt.plot(time, energies_numerical, label="Numerical Solution (Runge-Kutta)")
plt.plot(time, energies_analytic, label="Analytical Solution", linestyle="--")
plt.title("Energy Conservation: Numerical vs Analytical Solutions")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.legend()
plt.grid()
plt.show()