import numpy as np
import matplotlib.pyplot as plt

N = 12
K = 1.0
m = 1.0
dt = 0.01
T = 100

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

def rk4_method(dt, T, N, K, m):
    time_steps = int(T / dt)
    u = np.zeros((time_steps, N))
    v = np.zeros((time_steps, N))
    u[0, :] = 0
    v[0, 2] = 1

    def acceleration(u):
        acc = np.zeros(N)
        for j in range(N):
            left = u[j - 1] if j > 0 else 0
            right = u[j + 1] if j < N - 1 else 0
            acc[j] = K / m * (left - 2 * u[j] + right)
        return acc

    for t in range(1, time_steps):
        k1_u = v[t - 1]
        k1_v = acceleration(u[t - 1])

        k2_u = v[t - 1] + 0.5 * dt * k1_v
        k2_v = acceleration(u[t - 1] + 0.5 * dt * k1_u)

        k3_u = v[t - 1] + 0.5 * dt * k2_v
        k3_v = acceleration(u[t - 1] + 0.5 * dt * k2_u)

        k4_u = v[t - 1] + dt * k3_v
        k4_v = acceleration(u[t - 1] + dt * k3_u)

        u[t] = u[t - 1] + dt / 6 * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v[t] = v[t - 1] + dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return u, v

time_points = np.arange(0, T, dt)
u_num, v_num = rk4_method(dt, T, N, K, m)

u_analytic = np.zeros_like(u_num)
for t_idx, t in enumerate(time_points):
    for j in range(1, N + 1):
        u_analytic[t_idx, j - 1] = analytical_solution(t, j)

difference = np.abs(u_analytic - u_num)

max_differences_per_oscillator = np.max(np.abs(u_analytic - u_num), axis=0)
relative_errors_per_oscillator = max_differences_per_oscillator / np.max(np.abs(u_analytic), axis=0)

for j, (max_diff, rel_error) in enumerate(zip(max_differences_per_oscillator, relative_errors_per_oscillator), start=1):
    print(f"Oscillator {j}: Maximum Difference = {max_diff:.10f}, Relative Error = {rel_error:.10f}")