import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def newton_cooling_function(t, r, env_T, initial_T):
    return env_T+(initial_T-env_T)*np.exp(-r*t)

def calculate_r_seperatedly(t, T, env_T):
    r_values = []
    for i in range(len(t) - 1):
        delta_t = t[i + 1] - t[i]
        T_diff = T[i] - env_T
        next_T_diff = T[i + 1] - env_T
        r = -np.log(next_T_diff / T_diff) / delta_t
        r_values.append(r)
    return np.mean(r_values), np.std(r_values)

env_T = 17

t_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

t_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                       45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

initial_T_black = T_black[0]
initial_T_cream = T_cream[0]


params_black, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_black),
                            t_black, T_black)
r_black = params_black[0]

params_cream, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_cream),
                            t_cream, T_cream)
r_cream = params_cream[0]

print("==========Least Squares Methods==========")
print(f"black coffee's r: {r_black} min^-1")
print(f"cream coffee's r: {r_cream} min^-1")

r_black_mean, r_black_std = calculate_r_seperatedly(t_black, T_black, env_T)
r_cream_mean, r_cream_std = calculate_r_seperatedly(t_cream, T_cream, env_T)

print("==========Calculate partially==========")
print("black coffee's r: mean = {:.4f} min^-1, std = {:.4f} min^-1".format(r_black_mean, r_black_std))
print("cream coffee's r: mean = {:.4f} min^-1, std = {:.4f} min^-1".format(r_cream_mean, r_cream_std))

