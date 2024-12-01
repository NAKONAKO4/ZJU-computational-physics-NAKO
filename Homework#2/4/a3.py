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

# Step 3: Define u_j(t) and v_j(t)
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
print(v_j(1,3))
# Define total energy calculation for the system
def total_energy_system(t):
    """Compute total energy of the system at time t."""
    total_kinetic_energy = 0
    total_potential_energy = 0

    # Compute kinetic energy
    for j in range(1, N + 1):
        total_kinetic_energy += 0.5 * m * v_j(t, j) ** 2

    # Compute potential energy
    for j in range(1, N):  # Potential energy involves adjacent oscillators
        displacement_diff = u_j(t, j + 1) - u_j(t, j)
        total_potential_energy += 0.5 * K * displacement_diff ** 2

    return total_kinetic_energy + total_potential_energy

# Simulate over time to check energy conservation
time_steps = np.linspace(0, 100, 500)  # Simulate from t=0 to t=10 with 500 points
energies = [total_energy_system(t) for t in time_steps]

# Plot the total energy over time
plt.figure(figsize=(10, 6))
plt.plot(time_steps, energies, label="Total Energy")
plt.title("Energy Conservation in 12 Coupled Oscillators")
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.legend()
plt.grid()
plt.show()