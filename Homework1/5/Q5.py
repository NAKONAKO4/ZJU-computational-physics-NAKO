import numpy as np
import matplotlib.pyplot as plt

# 参数定义
gamma = 0.4
omega0 = 3.0
omega = 2.0
A0 = 3.1
x0, v0 = 1.0, 0.0  # 初始条件
t_end = 50
dt = 0.01

# 计算加速度的函数
def acceleration(x, v, t, A0, omega, omega0, gamma):
    return A0 * np.cos(omega * t) - 2 * gamma * v - omega0**2 * x

# 四阶 RK4 方法实现
def rk4_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt):
    t = np.arange(0, t_end, dt)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    x[0], v[0] = x0, v0

    for i in range(1, len(t)):
        k1_v = acceleration(x[i-1], v[i-1], t[i-1], A0, omega, omega0, gamma)
        k1_x = v[i-1]

        k2_v = acceleration(x[i-1] + 0.5 * dt * k1_x, v[i-1] + 0.5 * dt * k1_v, t[i-1] + 0.5 * dt, A0, omega, omega0, gamma)
        k2_x = v[i-1] + 0.5 * dt * k1_v

        k3_v = acceleration(x[i-1] + 0.5 * dt * k2_x, v[i-1] + 0.5 * dt * k2_v, t[i-1] + 0.5 * dt, A0, omega, omega0, gamma)
        k3_x = v[i-1] + 0.5 * dt * k2_v

        k4_v = acceleration(x[i-1] + dt * k3_x, v[i-1] + dt * k3_v, t[i-1] + dt, A0, omega, omega0, gamma)
        k4_x = v[i-1] + dt * k3_v

        x[i] = x[i-1] + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v[i] = v[i-1] + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return t, x, v

# 能量计算函数
def total_energy(x, v, omega0):
    return 0.5 * v**2 + 0.5 * omega0**2 * x**2

# 使用 RK4 方法计算 x(t), v(t) 和能量
t, x, v = rk4_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
E = total_energy(x, v, omega0)

# 绘制能量变化 ΔE_n = E_n - E_0
E_0 = E[0]
delta_E = E - E_0

plt.figure(figsize=(12, 6))
plt.plot(t, delta_E, label=r"$\Delta E_n = E_n - E_0$", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Baseline ($\Delta E_n = 0$)")
plt.xlabel("Time (s)")
plt.ylabel("Energy Difference $\Delta E_n$")
plt.title("Energy Difference $\Delta E_n$ vs Time (RK4 Method)")
plt.legend()
plt.grid()
plt.show()