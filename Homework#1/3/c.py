import numpy as np
import matplotlib.pyplot as plt

def exact_solution(t):
    x_exact = 1.1 * np.cos(t)
    v_exact = -1.1 * np.sin(t)
    return x_exact, v_exact

def euler_oscillator(delta_t, t_end):
    t = np.arange(0, t_end + delta_t, delta_t)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = 1.1, 0
    for i in range(1, len(t)):
        v[i] = v[i-1] - delta_t * x[i-1]
        x[i] = x[i-1] + delta_t * v[i-1]
    return t, x, v

def euler_cromer_oscillator(delta_t, t_end):
    t = np.arange(0, t_end + delta_t, delta_t)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = 1.1, 0
    for i in range(1, len(t)):
        v[i] = v[i-1] - delta_t * x[i-1]
        x[i] = x[i-1] + delta_t * v[i]
    return t, x, v

def euler_richardson_oscillator(delta_t, t_end):
    t = np.arange(0, t_end + delta_t, delta_t)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = 1.1, 0
    for i in range(1, len(t)):
        x_mid = x[i-1] + 0.5 * delta_t * v[i-1]
        v_mid = v[i-1] - 0.5 * delta_t * x[i-1]
        v[i] = v[i-1] - delta_t * x_mid
        x[i] = x[i-1] + delta_t * v_mid
    return t, x, v

def compute_oscillator_error(method, delta_t, t_end):
    t, x, v = method(delta_t, t_end)
    x_exact, v_exact = exact_solution(t)
    error_x = np.max(np.abs(x - x_exact))
    error_v = np.max(np.abs(v - v_exact))
    return error_x, error_v

t_end = 4*np.pi
delta_ts = [0.1, 0.05, 0.025, 0.01, 0.005]
errors = []

for delta_t in delta_ts:
    errors.append([
        delta_t,
        *compute_oscillator_error(euler_oscillator, delta_t, 2*t_end),
        *compute_oscillator_error(euler_cromer_oscillator, delta_t, 2*t_end),
        *compute_oscillator_error(euler_richardson_oscillator, delta_t, 2*t_end)
    ])

import pandas as pd
columns = ["Δt",
           "Euler (x error)", "Euler (v error)",
           "Euler-Cromer (x error)", "Euler-Cromer (v error)",
           "Euler-Richardson (x error)", "Euler-Richardson (v error)"]
error_df = pd.DataFrame(errors, columns=columns)
error_df.to_csv("harmonic_oscillator_error_table.csv", index=False)

# 绘制数值解与解析解的比较
delta_t = 0.1  # 选择较小步长
t_exact = np.linspace(0, t_end, 1000)
x_exact, v_exact = exact_solution(t_exact)

t_euler, x_euler, v_euler = euler_oscillator(delta_t, t_end)
t_cromer, x_cromer, v_cromer = euler_cromer_oscillator(delta_t, t_end)
t_richardson, x_richardson, v_richardson = euler_richardson_oscillator(delta_t, t_end)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_exact, x_exact, label="Exact Solution", color="black")
plt.plot(t_euler, x_euler, 'o-', label="Euler", markersize=4)
plt.plot(t_cromer, x_cromer, 's-', label="Euler-Cromer", markersize=4)
plt.plot(t_richardson, x_richardson, '^-', label="Euler-Richardson", markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Position (x)")
plt.legend()
plt.title("Position vs Time for Harmonic Oscillator")

plt.subplot(2, 1, 2)
plt.plot(t_exact, v_exact, label="Exact Solution", color="black")
plt.plot(t_euler, v_euler, 'o-', label="Euler", markersize=4)
plt.plot(t_cromer, v_cromer, 's-', label="Euler-Cromer", markersize=4)
plt.plot(t_richardson, v_richardson, '^-', label="Euler-Richardson", markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (v)")
plt.legend()
plt.title("Velocity vs Time for Harmonic Oscillator")
plt.tight_layout()
plt.show()