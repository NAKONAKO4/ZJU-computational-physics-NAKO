import numpy as np

def calculate_r_seperatedly(t, T, env_T):
    r_values = []
    for i in range(len(t) - 1):
        delta_t = t[i + 1] - t[i]
        T_diff = T[i] - env_T
        next_T_diff = T[i + 1] - env_T
        r = -np.log(next_T_diff / T_diff) / delta_t
        r_values.append(r)
    return np.mean(r_values)

t_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                    51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

t_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                    45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

env_T = 17

t_black_large_step = t_black[::2]
T_black_large_step = T_black[::2]
print(T_black_large_step)

t_black_small_step = np.linspace(0, 46, 47)
T_black_small_step = np.interp(t_black_small_step, t_black, T_black)

t_cream_large_step = t_cream[::2]
T_cream_large_step = T_cream[::2]

t_cream_small_step = np.linspace(0, 46, 47)
T_cream_small_step = np.interp(t_cream_small_step, t_cream, T_cream)

r_black_original = calculate_r_seperatedly(t_black, T_black, env_T)
r_black_large_step = calculate_r_seperatedly(t_black_large_step, T_black_large_step, env_T)
r_black_small_step = calculate_r_seperatedly(t_black_small_step, T_black_small_step, env_T)

r_cream_original = calculate_r_seperatedly(t_cream, T_cream, env_T)
r_cream_large_step = calculate_r_seperatedly(t_cream_large_step, T_cream_large_step, env_T)
r_cream_small_step = calculate_r_seperatedly(t_cream_small_step, T_cream_small_step, env_T)

relative_error_black_large = abs(r_black_large_step - r_black_original) / r_black_original * 100
relative_error_black_small = abs(r_black_small_step - r_black_original) / r_black_original * 100

relative_error_cream_large = abs(r_cream_large_step - r_cream_original) / r_cream_original * 100
relative_error_cream_small = abs(r_cream_small_step - r_cream_original) / r_cream_original * 100

print("黑咖啡冷却常数 r:")
print(f"  原始步长: r = {r_black_original:.4f}")
print(f"  较大步长: r = {r_black_large_step:.4f} (相对误差: {relative_error_black_large:.2f}%)")
print(f"  较小步长: r = {r_black_small_step:.4f} (相对误差: {relative_error_black_small:.2f}%)")

print("\n奶咖啡冷却常数 r:")
print(f"  原始步长: r = {r_cream_original:.4f}")
print(f"  较大步长: r = {r_cream_large_step:.4f} (相对误差: {relative_error_cream_large:.2f}%)")
print(f"  较小步长: r = {r_cream_small_step:.4f} (相对误差: {relative_error_cream_small:.2f}%)")