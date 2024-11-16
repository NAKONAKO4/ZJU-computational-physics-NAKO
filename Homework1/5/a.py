import numpy as np
import matplotlib.pyplot as plt

# 振子参数
gamma = 0.4  # 阻尼系数
omega0 = 3.0  # 自由振动频率
omega = 2.0  # 外加驱动频率
A0 = 3.1  # 驱动振幅

# 初始条件
x0, v0 = 1.0, 0.0  # 初始位置和速度
t_end = 50  # 模拟时间
dt = 0.01  # 时间步长

# 振动方程数值求解 (欧拉方法)
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

# 计算 x(t) 并绘图
t, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)

plt.figure(figsize=(12, 6))
plt.plot(t, x, label=r"$A_0=3.1$", color="blue")
plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("Driven Oscillator: Displacement x(t) vs Time")
plt.legend()
plt.grid()
plt.show()

# 新初始条件
x0_new, v0_new = 0.5, 1.0

# 计算 x(t) 并绘图
t, x_new, _ = driven_oscillator(A0, omega, omega0, gamma, x0_new, v0_new, t_end, dt)

plt.figure(figsize=(12, 6))
plt.plot(t, x, label=r"Initial: $x(0)=1.0, \dot{x}(0)=0.0$", color="blue")
plt.plot(t, x_new, label=r"Initial: $x(0)=0.5, \dot{x}(0)=1.0$", color="orange")
plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("Driven Oscillator with Different Initial Conditions")
plt.legend()
plt.grid()
plt.show()

# 计算总能量
def total_energy(x, v, omega0):
    return 0.5 * v**2 + 0.5 * omega0**2 * x**2

# 能量计算
t, x, v = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
E = total_energy(x, v, omega0)

plt.figure(figsize=(12, 6))
plt.plot(t, E, label="Total Energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy $E_n$")
plt.title("Total Energy vs Time")
plt.legend()
plt.grid()
plt.show()

# 不同的 ω_0 和 ω
omega0_values = [2.8, 3.0, 3.2]
omega_values = [2.0, 2.5, 3.0]

plt.figure(figsize=(12, 8))

for omega0 in omega0_values:
    for omega in omega_values:
        t, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
        plt.plot(t, x, label=rf"$\omega_0={omega0}, \omega={omega}$")

plt.xlabel("Time (s)")
plt.ylabel("Displacement x(t)")
plt.title("Driven Oscillator with Different Frequencies")
plt.legend()
plt.grid()
plt.show()

# 相空间轨迹
plt.figure(figsize=(12, 6))
plt.plot(x, v, label="Phase Space Trajectory")
plt.xlabel("Displacement x(t)")
plt.ylabel("Velocity v(t)")
plt.title("Phase Space Trajectory")
plt.legend()
plt.grid()
plt.show()

# 振幅和相位
def amplitude_phase(A0, omega, omega0, gamma, t_end, dt):
    t, x, _ = driven_oscillator(A0, omega, omega0, gamma, x0, v0, t_end, dt)
    amplitude = np.max(x[int(len(t)/2):])  # 取稳态后的最大值
    phase = np.arctan2(-gamma * omega, omega0**2 - omega**2)
    return amplitude, phase

# 不同 ω 的振幅和相位
omega_values = np.linspace(1.0, 4.0, 20)
amplitudes, phases = [], []

for omega in omega_values:
    amp, phase = amplitude_phase(A0, omega, omega0, gamma, t_end, dt)
    amplitudes.append(amp)
    phases.append(phase)

plt.figure(figsize=(12, 6))
plt.plot(omega_values, amplitudes, label="Amplitude A(ω)")
plt.xlabel("Driving Frequency ω")
plt.ylabel("Amplitude A(ω)")
plt.title("Amplitude vs Driving Frequency")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(omega_values, phases, label="Phase Difference δ(ω)")
plt.xlabel("Driving Frequency ω")
plt.ylabel("Phase Difference δ(ω)")
plt.title("Phase Difference vs Driving Frequency")
plt.grid()
plt.legend()
plt.show()