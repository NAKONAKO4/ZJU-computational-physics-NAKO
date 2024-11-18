import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_oscillator(t, omega0, gamma, omega, A0):
    dt = t[1] - t[0]
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    for i in range(1, len(t)):
        a = A0 * np.cos(omega * t[i - 1]) - 2 * gamma * v[i - 1] - omega0 ** 2 * x[i - 1]
        v[i] = v[i - 1] + a * dt
        x[i] = x[i - 1] + v[i] * dt

    return x, v


t = np.linspace(0, 50, 5000)
gamma = 0.4
A0 = 3.1

params = [
    {"omega0": 3.0, "omega": 2.0},
    {"omega0": 4.0, "omega": 3.0},
    {"omega0": 5.0, "omega": 4.0}
]

results = []

for p in params:
    x, _ = compute_oscillator(t, p["omega0"], gamma, p["omega"], A0)
    results.append({"params": p, "x": x})

#Q6
periods = []
for i, result in enumerate(results):
    p = result["params"]
    x = result["x"]

    steady_state = x[int(len(x) * 0.8):]
    time_steady = t[int(len(x) * 0.8):]

    peaks = np.where((steady_state[1:-1] > steady_state[:-2]) & (steady_state[1:-1] > steady_state[2:]))[0] + 1
    peak_times = time_steady[peaks]

    if len(peak_times) > 1:
        period = np.mean(np.diff(peak_times))
    else:
        period = np.nan

    periods.append(period)
    angular_frequency = 2 * np.pi / period if period else np.nan

    print(f"Case {i + 1} (ω₀={p['omega0']}, ω={p['omega']}):")
    print(f"  Period (T): {period:.4f}")
    print(f"  Angular Frequency (ω): {angular_frequency:.4f}")
    print()

#loglog
frequencies = [result["params"]["omega"] for result in results]
valid_periods = [p for p in periods if not np.isnan(p)]

plt.figure(figsize=(10, 6))
plt.loglog(frequencies[:len(valid_periods)], valid_periods, 'o-', label="T ~ ω^α")
for freq, period in zip(frequencies[:len(valid_periods)], valid_periods):
    plt.annotate(f"({freq:.4f}, {period:.4f})", xy=(freq, period),
                 xytext=(5, 5), textcoords="offset points", fontsize=8, color="darkred")
plt.xlabel("Angular Frequency ω")
plt.ylabel("Period T")
plt.title("Log-Log Plot of T vs ω")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()
log_frequencies = np.log(frequencies)
log_periods = np.log(valid_periods)

slope, intercept, r_value, p_value, std_err = linregress(log_frequencies, log_periods)



plt.xlabel("Angular Frequency ω")
plt.ylabel("Period T")
plt.title("Log-Log Plot of T vs ω with Fitted Line")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()

# Print results
print(f"Slope (α): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")