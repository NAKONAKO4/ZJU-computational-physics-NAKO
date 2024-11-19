import numpy as np
import matplotlib.pyplot as plt


def driven_oscillator(t, omega0, gamma, omega, A0):
    dt = t[1] - t[0]
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    for i in range(1, len(t)):
        a = A0 * np.cos(omega * t[i - 1]) - 2 * gamma * v[i - 1] - omega0 ** 2 * x[i - 1]
        v[i] = v[i - 1] + a * dt
        x[i] = x[i - 1] + v[i] * dt

    return x


t = np.linspace(0, 100, 10000)  # 时间数组
omega0 = 3.0
A0 = 3.1
gamma_values = [0.5, 2.0]
omega_values = [0.0, 1.0, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8]

# Q9: 计算 A(ω) 和 δ(ω)
results = {}
for gamma in gamma_values:
    A_omega = []
    delta_omega = []

    for omega in omega_values:
        x = driven_oscillator(t, omega0, gamma, omega, A0)
        steady_state = x[int(len(x) * 0.8):]  # 提取稳态部分
        amplitude = np.max(np.abs(steady_state))  # 振幅

        # 计算相位差 δ(ω)
        applied_force = A0 * np.cos(omega * t[int(len(x) * 0.8):])
        delta = -np.arccos(np.correlate(steady_state, applied_force) /
                          (np.linalg.norm(steady_state) * np.linalg.norm(applied_force)))

        A_omega.append(amplitude)
        delta_omega.append(delta[0] if isinstance(delta, np.ndarray) else delta)

    results[gamma] = {"A_omega": A_omega, "delta_omega": delta_omega}

# Q9: 绘制 A(ω) 和 δ(ω) 图像
for gamma in gamma_values:
    plt.figure(figsize=(10, 6))

    # A(ω) 图
    plt.subplot(2, 1, 1)
    plt.plot(omega_values, results[gamma]["A_omega"], 'o-', label=f"γ = {gamma}")
    plt.xlabel("Angular Frequency ω")
    plt.ylabel("Amplitude A(ω)")
    plt.title(f"Amplitude vs Angular Frequency (γ = {gamma})")
    plt.grid()
    plt.legend()

    # δ(ω) 图
    plt.subplot(2, 1, 2)
    plt.plot(omega_values, results[gamma]["delta_omega"], 'o-', label=f"γ = {gamma}")
    plt.xlabel("Angular Frequency ω")
    plt.ylabel("Phase Difference δ(ω)")
    plt.title(f"Phase Difference vs Angular Frequency (γ = {gamma})")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Q10: 找到 A(ω) 的最大值和对应的 ω_m
for gamma in gamma_values:
    A_omega = results[gamma]["A_omega"]
    max_index = np.argmax(A_omega)
    omega_m = omega_values[max_index]
    print(f"γ = {gamma}: Maximum A(ω) occurs at ω_m = {omega_m:.2f} (natural ω₀ = {omega0})")