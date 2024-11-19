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


t, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
t1, x1, _ = driven_oscillator(0, omega, omega0, gamma, x0, v0, t_end, dt)
plt.figure(figsize=(12, 6))
plt.plot(t, x, label=r"$A_0=3.1$", color="blue")
plt.plot(t, x1, label=r"$A_0=0$", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("x(t) vs Time")
plt.legend()
plt.grid()
plt.show()

x0_new, v0_new = 0.5, 1.0

t, x_new, _ = driven_oscillator(A0, omega, omega0, gamma, x0_new, v0_new, t_end, dt)
t1, x_new1, _ = driven_oscillator(0, omega, omega0, gamma, x0_new, v0_new, t_end, dt)
plt.figure(figsize=(12, 6))
plt.plot(t, x, label=r"Initial: $x(0)=1.0, \dot{x}(0)=0.0$", color="blue")
plt.plot(t, x_new, label=r"Initial: $x(0)=0.5, \dot{x}(0)=1.0$", color="orange")
#plt.plot(t1,x_new1, label=r"Initial: $x(0)=0.5, \dot{x}(0)=1.0, A_0=0$")
plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("Driven Oscillator with Different Initial Conditions")
plt.legend()
plt.grid()
plt.show()



