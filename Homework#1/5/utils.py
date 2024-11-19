import numpy as np
def driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt):
    t = np.arange(0, t_end, dt)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0

    for i in range(1, len(t)):
        a = A0 * np.cos(omega * t[i-1]) - 2 * gamma * v[i-1] - omega0**2 * x[i-1]
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt

    return t, x, v