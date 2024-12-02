import numpy as np

N = 12
K = 1.0
m = 1.0
dt = 0.01
T = 10

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

max_differences_per_oscillator = np.max(np.abs(u_analytic - u_num), axis=0)
relative_errors_per_oscillator = max_differences_per_oscillator / np.max(np.abs(u_analytic), axis=0)

for j, (max_diff, rel_error) in enumerate(zip(max_differences_per_oscillator, relative_errors_per_oscillator), start=1):
    print(f"Oscillator {j}: Maximum Difference = {max_diff:.6f}, Relative Error = {rel_error:.6f}")