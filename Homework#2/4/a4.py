import numpy as np
import matplotlib.pyplot as plt

# System parameters
N = 12  # Number of oscillators
K = 1.0  # Coupling constant
m = 1.0  # Mass of oscillators

# Step 1: Compute frequencies and modes
frequencies = np.array([2 * np.sqrt(K / m) * np.sin((k * np.pi) / (2 * (N + 1))) for k in range(1, N + 1)])
modes = np.array([[np.sin((k * j * np.pi) / (N + 1)) for j in range(1, N + 1)] for k in range(1, N + 1)])

# Step 2: Compute amplitudes A_k based on initial conditions
# Initial condition: v_3(0) = 1, all other initial velocities = 0
A_k = np.array([
    (2 / (N + 1)) * np.sum(
        [np.sin((k * j * np.pi) / (N + 1)) * (1 if j == 3 else 0) for j in range(1, N + 1)]
    ) / frequencies[k - 1]
    for k in range(1, N + 1)
])

# Define u_j(t) and v_j(t)
def u_j(t, j):
    """Compute displacement u_j(t) for oscillator j at time t."""
    return np.sum([
        A_k[k] * modes[k, j - 1] * np.sin(frequencies[k] * t)
        for k in range(N)
    ])

def v_j(t, j):
    """Compute velocity v_j(t) for oscillator j at time t."""
    return np.sum([
        A_k[k] * modes[k, j - 1] * frequencies[k] * np.cos(frequencies[k] * t)
        for k in range(N)
    ])

# Simulate vibration process
time_points = np.linspace(0, 40, 500)  # Simulate from t=0 to t=10 with 500 points
displacement_data = np.zeros((len(time_points), N))  # Rows: time, Columns: oscillators

for t_idx, t in enumerate(time_points):
    for j in range(1, N + 1):
        displacement_data[t_idx, j - 1] = u_j(t, j)

# Plot the vibration process
plt.figure(figsize=(12, 8))
for j in range(1, N + 1):
    plt.plot(time_points, displacement_data[:, j - 1], label=f'Oscillator {j}')

plt.title("Vibration of 12 Coupled Oscillators")
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.legend(loc='upper right', fontsize='small')
plt.grid()
plt.show()