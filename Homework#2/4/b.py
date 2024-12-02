import numpy as np
import matplotlib.pyplot as plt

N = 12
K = 1.0
m = 1.0
dt = 0.01
T = 15

frequencies = np.array([2 * np.sqrt(K / m) * np.sin((k * np.pi) / (2 * (N + 1))) for k in range(1, N + 1)])
modes = np.array([[np.sin((k * j * np.pi) / (N + 1)) for j in range(1, N + 1)] for k in range(1, N + 1)])
A_k = np.array([
    (2 / (N + 1)) * np.sum(
        [np.sin((k * j * np.pi) / (N + 1)) * (1 if j == 3 else 0) for j in range(1, N + 1)]
    ) / frequencies[k - 1]
    for k in range(1, N + 1)
])

def analytical_solution(t, j):
    return np.sum([
        A_k[k] * modes[k, j - 1] * np.sin(frequencies[k] * t)
        for k in range(N)
    ])


def euler_method(dt, T, N, K, m):
    time_steps = int(T / dt)
    u = np.zeros((time_steps, N))
    v = np.zeros((time_steps, N))
    u[0, :] = 0
    v[0, 2] = 1

    for t in range(1, time_steps):
        for j in range(N):
            left = u[t - 1, j - 1] if j > 0 else 0
            right = u[t - 1, j + 1] if j < N - 1 else 0
            a_j = K / m * (left - 2 * u[t - 1, j] + right)
            v[t, j] = v[t - 1, j] + dt * a_j
            u[t, j] = u[t - 1, j] + dt * v[t - 1, j]

    return u, v

time_points = np.arange(0, T, dt)
u_num, v_num = euler_method(dt, T, N, K, m)

u_analytic = np.zeros_like(u_num)
for t_idx, t in enumerate(time_points):
    for j in range(1, N + 1):
        u_analytic[t_idx, j - 1] = analytical_solution(t, j)
def compute_total_energy(u, v, K, m):
    kinetic_energy = 0.5 * m * np.sum(v**2, axis=1)
    potential_energy = 0.5 * K * np.sum((np.diff(u, axis=1))**2, axis=1)
    return kinetic_energy + potential_energy

energy_values = compute_total_energy(u_num, v_num, K, m)

plt.figure(figsize=(10, 6))
plt.plot(time_points, energy_values, label="Total Energy (Euler Method)")
plt.title("Energy Conservation Using Euler Method")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.legend()
plt.grid()
plt.show()

def compute_energy_analytic(u_analytic, K, m, time_points):
    v_analytic = np.zeros_like(u_analytic)
    for t_idx, t in enumerate(time_points):
        for j in range(1, N + 1):
            v_analytic[t_idx, j - 1] = np.sum([
                A_k[k] * modes[k, j - 1] * frequencies[k] * np.cos(frequencies[k] * t)
                for k in range(N)
            ])
    kinetic_energy = 0.5 * m * np.sum(v_analytic**2, axis=1)
    potential_energy = 0.5 * K * np.sum((np.diff(u_analytic, axis=1))**2, axis=1)
    return kinetic_energy + potential_energy

analytic_energy_values = compute_energy_analytic(u_analytic, K, m, time_points)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_points, analytic_energy_values, label="Total Energy (Analytical Solution)")
plt.title("Energy Conservation Using Analytical Solution")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.legend()
plt.grid()
plt.show()