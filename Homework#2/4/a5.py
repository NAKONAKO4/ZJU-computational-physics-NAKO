import numpy as np
import matplotlib.pyplot as plt

# System parameters
N = 12  # Number of oscillators
K = 1.0  # Coupling constant
m = 1.0  # Mass of oscillators
dt = 0.01  # Time step
T = 10  # Total simulation time

# Step 1: Compute frequencies and modes for analytical solution
frequencies = np.array([2 * np.sqrt(K / m) * np.sin((k * np.pi) / (2 * (N + 1))) for k in range(1, N + 1)])
modes = np.array([[np.sin((k * j * np.pi) / (N + 1)) for j in range(1, N + 1)] for k in range(1, N + 1)])
A_k = np.array([
    (2 / (N + 1)) * np.sum(
        [np.sin((k * j * np.pi) / (N + 1)) * (1 if j == 3 else 0) for j in range(1, N + 1)]
    ) / frequencies[k - 1]
    for k in range(1, N + 1)
])

def analytical_solution(t, j):
    """Compute analytical displacement u_j(t)."""
    return np.sum([
        A_k[k] * modes[k, j - 1] * np.sin(frequencies[k] * t)
        for k in range(N)
    ])

# Step 2: Define numerical solution using Euler method
def euler_method(dt, T, N, K, m):
    """Compute numerical solution using Euler method."""
    time_steps = int(T / dt)
    u = np.zeros((time_steps, N))  # Displacements
    v = np.zeros((time_steps, N))  # Velocities
    u[0, :] = 0  # Initial displacements
    v[0, 2] = 1  # Initial velocity at oscillator 3

    for t in range(1, time_steps):
        # Update velocities and displacements
        for j in range(N):
            # Compute acceleration
            left = u[t - 1, j - 1] if j > 0 else 0
            right = u[t - 1, j + 1] if j < N - 1 else 0
            a_j = K / m * (left - 2 * u[t - 1, j] + right)
            v[t, j] = v[t - 1, j] + dt * a_j
            u[t, j] = u[t - 1, j] + dt * v[t - 1, j]

    return u, v

# Compute numerical solution
time_points = np.arange(0, T, dt)
u_num, v_num = euler_method(dt, T, N, K, m)

# Step 3: Compute analytical solution and differences
u_analytic = np.zeros_like(u_num)
for t_idx, t in enumerate(time_points):
    for j in range(1, N + 1):
        u_analytic[t_idx, j - 1] = analytical_solution(t, j)

# Compute the difference between analytical and numerical solutions
difference = np.abs(u_analytic - u_num)

# Step 4: Plot results
plt.figure(figsize=(12, 8))
for j in range(1, N + 1):
    plt.plot(time_points, difference[:, j - 1], label=f'Oscillator {j}')

plt.title("Difference Between Analytical and Numerical Solutions")
plt.xlabel("Time")
plt.ylabel("Difference in Displacement")
plt.legend(loc='upper right', fontsize='small')
plt.grid()
plt.show()