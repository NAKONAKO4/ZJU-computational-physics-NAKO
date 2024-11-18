import numpy as np
import matplotlib.pyplot as plt
from utils import driven_oscillator

omega0 = 3
gamma = 0.5
A0 = 3.1
gamma_values = [0.5, 2.0]
omega_values = [0.5, 1.0, 2.0, 2.8, 3.0]
x0=1.0
v0=0.0
t_end = 500
dt=0.1
A_omega = []
delta_omega = []

for omega in omega_values:
    t, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
    steady_state = x[int(len(x) * 0.8):]
    time_steady = t[int(len(x) * 0.8):]

    amplitude = (np.max(steady_state) - np.min(steady_state)) / 2
    A_omega.append(amplitude)

    applied_force = A0 * np.cos(omega * time_steady)
    correlation = np.dot(steady_state, applied_force) / (np.linalg.norm(steady_state) * np.linalg.norm(applied_force))
    delta = -np.arccos(np.clip(correlation, -1, 1))
    delta_omega.append(delta)

    reconstructed_x = amplitude * np.cos(omega * time_steady + delta)

    plt.figure(figsize=(10, 4))
    plt.plot(time_steady[:100], steady_state[:100], label="Numerical $x(t)$", color="blue")
    plt.plot(time_steady[:100], reconstructed_x[:100], '--', label="Reconstructed $x(t)$", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.title(f"Comparison of $x(t)$ and Reconstructed $x(t)$ (ω = {omega:.2f})")
    plt.legend()
    plt.grid()
    plt.show()




omega_values = [0.0, 1.0, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8]

results = {}
for gamma in gamma_values:
    A_omega = []
    delta_omega = []

    for omega in omega_values:
        _, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
        steady_state = x[int(len(x) * 0.8):]
        amplitude = np.max(np.abs(steady_state))

        applied_force = A0 * np.cos(omega * t[int(len(x) * 0.8):])
        delta = -np.arccos(np.correlate(steady_state, applied_force) /
                          (np.linalg.norm(steady_state) * np.linalg.norm(applied_force)))

        A_omega.append(amplitude)
        delta_omega.append(delta[0] if isinstance(delta, np.ndarray) else delta)

    results[gamma] = {"A_omega": A_omega, "delta_omega": delta_omega}

for gamma in gamma_values:
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(omega_values, results[gamma]["A_omega"], 'o-', label=f"γ = {gamma}")
    plt.xlabel("Angular Frequency ω")
    plt.ylabel("Amplitude A(ω)")
    plt.title(f"Amplitude vs Angular Frequency (γ = {gamma})")
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(omega_values, results[gamma]["delta_omega"], 'o-', label=f"γ = {gamma}")
    plt.xlabel("Angular Frequency ω")
    plt.ylabel("Phase Difference δ(ω)")
    plt.title(f"Phase Difference vs Angular Frequency (γ = {gamma})")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

for gamma in gamma_values:
    A_omega = results[gamma]["A_omega"]
    max_index = np.argmax(A_omega)
    omega_m = omega_values[max_index]
    print(f"γ = {gamma}: Maximum A(ω) occurs at ω_m = {omega_m:.2f} (natural ω₀ = {omega0})")