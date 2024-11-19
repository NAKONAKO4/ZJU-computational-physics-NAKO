import numpy as np
from scipy.optimize import curve_fit

def newton_cooling_function(t, r, env_T, initial_T):
    return env_T + (initial_T - env_T) * np.exp(-r * t)

t_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                    51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

t_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                    45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

env_T = 17
initial_T_black = T_black[0]
initial_T_cream = T_cream[0]

params_black, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_black), t_black, T_black)
r_black_original = params_black[0]

params_cream, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_cream), t_cream, T_cream)
r_cream_original = params_cream[0]

t_black_large_step = t_black[::2]
T_black_large_step = T_black[::2]
params_black_large, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_black), t_black_large_step, T_black_large_step)
r_black_large_step = params_black_large[0]

t_cream_large_step = t_cream[::2]
T_cream_large_step = T_cream[::2]
params_cream_large, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_cream), t_cream_large_step, T_cream_large_step)
r_cream_large_step = params_cream_large[0]

t_black_small_step = np.linspace(0, 46, 47)
T_black_small_step = np.interp(t_black_small_step, t_black, T_black)
params_black_small, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_black), t_black_small_step, T_black_small_step)
r_black_small_step = params_black_small[0]

t_cream_small_step = np.linspace(0, 46, 47)
T_cream_small_step = np.interp(t_cream_small_step, t_cream, T_cream)
params_cream_small, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_cream), t_cream_small_step, T_cream_small_step)
r_cream_small_step = params_cream_small[0]

relative_error_black_large = abs(r_black_large_step - r_black_original) / r_black_original * 100
relative_error_black_small = abs(r_black_small_step - r_black_original) / r_black_original * 100

relative_error_cream_large = abs(r_cream_large_step - r_cream_original) / r_cream_original * 100
relative_error_cream_small = abs(r_cream_small_step - r_cream_original) / r_cream_original * 100

print("Black coffee cooling constant r:")
print(f"  Original step: r = {r_black_original:.6f}")
print(f"  Larger step: r = {r_black_large_step:.6f} (Relative error: {relative_error_black_large:.2f}%)")
print(f"  Smaller step: r = {r_black_small_step:.6f} (Relative error: {relative_error_black_small:.2f}%)")

print("\nCream coffee cooling constant r:")
print(f"  Original step: r = {r_cream_original:.6f}")
print(f"  Larger step: r = {r_cream_large_step:.6f} (Relative error: {relative_error_cream_large:.2f}%)")
print(f"  Smaller step: r = {r_cream_small_step:.6f} (Relative error: {relative_error_cream_small:.2f}%)")