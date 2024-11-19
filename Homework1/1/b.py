import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def newton_cooling_function(t, r, env_T, initial_T):
    return env_T+(initial_T-env_T)*np.exp(-r*t)

t_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                    51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
t_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
T_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                    45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])
env_T = 17
initial_T_black = T_black[0]
initial_T_cream = T_cream[0]

params_black, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_black),
                            t_black, T_black)
r_black = params_black[0]

params_cream, _ = curve_fit(lambda t, r: newton_cooling_function(t, r, env_T, initial_T_cream),
                            t_cream, T_cream)
r_cream = params_cream[0]

time_fit = np.linspace(0, 46, 200)
temp_fit_black = newton_cooling_function(time_fit, r_black, env_T, initial_T_black)
temp_fit_cream = newton_cooling_function(time_fit, r_cream, env_T, initial_T_cream)

plt.figure(figsize=(12, 8))

plt.plot(t_black, T_black, 'o', label='Experimental Data (Black Coffee)')
plt.plot(time_fit, temp_fit_black, '-', label=f'Fitted Curve (Black Coffee, r = {r_black:.4f})')

plt.plot(t_cream, T_cream, 's', label='Experimental Data (Cream Coffee)')
plt.plot(time_fit, temp_fit_cream, '--', label=f'Fitted Curve (Cream Coffee, r = {r_cream:.4f})')

plt.xlabel('Time (minutes)', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.title('Cooling of Black Coffee and Cream Coffee', fontsize=16)
plt.legend()
plt.grid()
plt.show()