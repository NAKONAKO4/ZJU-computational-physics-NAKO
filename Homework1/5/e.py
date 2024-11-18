import numpy as np
import matplotlib.pyplot as plt
from utils import driven_oscillator

gamma = 0.4
omega0 = 3.0
omega = 2.8
A0 = 3.1

x0, v0 = 1.0, 0.0
t_end = 50
dt = 0.01

t, xp, vp = driven_oscillator(A0, omega, omega0, gamma, x0, 0.5, t_end, dt)
t2, x2, v2 = driven_oscillator(A0, omega, omega0, gamma, 1,1, t_end, dt)
#print(t,xp,vp)
plt.figure(figsize=(12, 6))
#plt.plot(x, v, label="Phase Space Trajectory")
plt.plot(xp, vp, label=r"Phase Space Trajectory1")
plt.plot(x2, v2, label=r"Phase Space Trajectory2")
plt.xlabel("Displacement x(t)")
plt.ylabel("Velocity v(t)")
plt.title("Phase Space Trajectory")
plt.legend()
plt.grid()
plt.show()