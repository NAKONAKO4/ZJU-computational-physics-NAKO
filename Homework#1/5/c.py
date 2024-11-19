import numpy as np
import matplotlib.pyplot as plt
from utils import driven_oscillator

gamma = 0.4
omega0 = 3.0
omega = 2.0
A0 = 3.1

x0, v0 = 1.0, 0.0
t_end = 50
dt = 0.01

def total_energy(x, v, omega0):
    return 0.5 * v**2 + 0.5 * omega0**2 * x**2

t, x, v = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
E = total_energy(x, v, omega0)
E_0 = E[0]
plt.figure(figsize=(12, 6))
plt.plot(t, E-E_0, label="Total Energy")
plt.axhline(0, color="red", linestyle="--", label="Baseline ($\Delta E_n = 0$)")
plt.xlabel("Time (s)")
plt.ylabel("Energy $E_n$")
plt.title("Total Energy vs Time")
plt.legend()
plt.grid()
plt.show()