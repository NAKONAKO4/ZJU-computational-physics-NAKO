import numpy as np
import matplotlib.pyplot as plt

# 参数定义
g = 9.8  # 重力加速度 (m/s^2)
y0 = 32  # 初始位置 (m)
v0 = 0   # 初始速度 (m/s)
t_end = 2  # 模拟时间 (s)
exact_y = lambda t: y0 + v0 * t - 0.5 * g * t**2  # 解析位置解
exact_v = lambda t: v0 - g * t  # 解析速度解

# 欧拉方法
def euler_method(delta_t):
    t = np.arange(0, t_end + delta_t, delta_t)
    v = np.zeros(len(t))
    y = np.zeros(len(t))
    y[0], v[0] = y0, v0
    for i in range(1, len(t)):
        v[i] = v[i-1] - g * delta_t
        y[i] = y[i-1] + v[i-1] * delta_t
    return t, y, v

# 欧拉-克罗默方法
def euler_cromer_method(delta_t):
    t = np.arange(0, t_end + delta_t, delta_t)
    v = np.zeros(len(t))
    y = np.zeros(len(t))
    y[0], v[0] = y0, v0
    for i in range(1, len(t)):
        v[i] = v[i-1] - g * delta_t
        y[i] = y[i-1] + v[i] * delta_t  # 更新速度后再更新位置
    return t, y, v

# 欧拉-理查森方法
def euler_richardson_method(delta_t):
    t = np.arange(0, t_end + delta_t, delta_t)
    v = np.zeros(len(t))
    y = np.zeros(len(t))
    y[0], v[0] = y0, v0
    for i in range(1, len(t)):
        v_mid = v[i-1] - 0.5 * g * delta_t
        y_mid = y[i-1] + 0.5 * v[i-1] * delta_t
        v[i] = v[i-1] - g * delta_t
        y[i] = y[i-1] + v_mid * delta_t
    return t, y, v

# 比较不同算法的误差
def compute_error(method, delta_t):
    t, y, v = method(delta_t)
    y_exact = exact_y(t)
    v_exact = exact_v(t)
    error_y = np.max(np.abs(y - y_exact))
    error_v = np.max(np.abs(v - v_exact))
    return error_y, error_v

# 测试不同步长
delta_ts = [0.2, 0.1, 0.05, 0.01, 0.005]
errors = []

for delta_t in delta_ts:
    errors.append([
        delta_t,
        *compute_error(euler_method, delta_t),
        *compute_error(euler_cromer_method, delta_t),
        *compute_error(euler_richardson_method, delta_t)
    ])

# 将误差表格显示出来
import pandas as pd
columns = ["Δt",
           "Euler (y error)", "Euler (v error)",
           "Euler-Cromer (y error)", "Euler-Cromer (v error)",
           "Euler-Richardson (y error)", "Euler-Richardson (v error)"]
error_df = pd.DataFrame(errors, columns=columns)
print(error_df)
error_df.to_csv("y&v_error_table.csv", index=False)
print("表格已保存为 'harmonic_oscillator_error_table.csv'")
#import ace_tools as tools; tools.display_dataframe_to_user(name="Comparison of Algorithms: Error Table", dataframe=error_df)

# 绘制位置和速度的图像
delta_t = 0.01  # 选择一个较小的时间步长
t_euler, y_euler, v_euler = euler_method(delta_t)
t_cromer, y_cromer, v_cromer = euler_cromer_method(delta_t)
t_richardson, y_richardson, v_richardson = euler_richardson_method(delta_t)
t_exact = np.arange(0, t_end, 0.001)
y_exact = exact_y(t_exact)
v_exact = exact_v(t_exact)

plt.figure(figsize=(16, 12))
# 位置图
plt.subplot(2, 1, 1)
plt.plot(t_exact, y_exact, label="Exact Solution", color="black")
plt.plot(t_euler, y_euler, 'o-', label="Euler", markersize=4)
plt.plot(t_cromer, y_cromer, 's-', label="Euler-Cromer", markersize=4)
plt.plot(t_richardson, y_richardson, '^-', label="Euler-Richardson", markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Position (y)")
plt.legend()
plt.title("Position vs Time")

# 速度图
plt.subplot(2, 1, 2)
plt.plot(t_exact, v_exact, label="Exact Solution", color="black")
plt.plot(t_euler, v_euler, label="Euler", markersize=4)
plt.plot(t_cromer, v_cromer,  label="Euler-Cromer", markersize=4)
plt.plot(t_richardson, v_richardson, label="Euler-Richardson", markersize=4)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (v)")
plt.legend()
plt.title("Velocity vs Time")
plt.tight_layout()
plt.show()