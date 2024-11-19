import numpy as np

def time_to_temperature(r, T_env, T_initial, T_target):
    return -np.log((T_target - T_env) / (T_initial - T_env)) / r

r = 0.0259
T_env = 17
T_initial_black = 82.3
T_targets = [49, 33, 25]

times = [time_to_temperature(r, T_env, T_initial_black, T_target) for T_target in T_targets]

for T_target, time in zip(T_targets, times):
    print(f"Cool down to {T_target}°C needed: {time:.2f} minutes")