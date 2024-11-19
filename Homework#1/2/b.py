import numpy as np
import matplotlib.pyplot as plt

r = 0.0259
T_s = 17
T_0 = 82.3
t_target = 1.60

def analytical_solution(t, T_s, T_0, r):
    return T_s + (T_0 - T_s) * np.exp(-r * t)

def euler_method(T_s, T_0, r, delta_t, t_target):
    time_points = np.arange(0, t_target + delta_t, delta_t)
    T = np.zeros(len(time_points))
    T[0] = T_0
    for n in range(1, len(time_points)):
        T[n] = T[n - 1] + delta_t * (-r * (T[n - 1] - T_s))
    return time_points, T

delta_ts = [0.1, 0.05, 0.025, 0.01, 0.005]
errors = []

for delta_t in delta_ts:
    _, T_numeric = euler_method(T_s, T_0, r, delta_t, t_target)
    T_exact = analytical_solution(t_target, T_s, T_0, r)
    error = abs(T_numeric[-1] - T_exact)
    errors.append(error)

plt.figure(figsize=(10, 6))
plt.loglog(delta_ts, errors, 'o-', label='Error vs. Δt')
plt.xlabel('Time Step Δt (log scale)', fontsize=12)
plt.ylabel('Error (log scale)', fontsize=12)
plt.title('Error vs. Time Step Size in Euler Method', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

print("b：")
for i, delta_t in enumerate(delta_ts):
    print(f"Δt = {delta_t:.3f}, Error = {errors[i]:.6f}")

def find_delta_t(T_s, T_0, r, t_target, tolerance):
    delta_t = 0.1
    while True:
        _, T_numeric = euler_method(T_s, T_0, r, delta_t, t_target)
        T_exact = analytical_solution(t_target, T_s, T_0, r)
        error = abs(T_numeric[-1] - T_exact)
        if error/T_exact <= tolerance:
            return delta_t, error
        delta_t /= 2

tolerance = 0.001
delta_t_1_60, error_1_60 = find_delta_t(T_s, T_0, r, 1.60, tolerance)
delta_t_5_5, error_5_5 = find_delta_t(T_s, T_0, r, 5.5, tolerance)

print("c：")
print(f"t = 1.60，Δt = {delta_t_1_60:.6f}, respective_error = {error_1_60:.6f}%")
print(f"t = 5.50，Δt = {delta_t_5_5:.6f}, respective_error = {error_5_5:.6f}%")