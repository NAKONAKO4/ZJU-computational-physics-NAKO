import numpy as np
import matplotlib.pyplot as plt

# 牛顿冷却模型参数
r = 0.0273  # 冷却常数
T_s = 17  # 环境温度
T_0 = 82.3  # 初始温度
t_target = 1.60  # 目标时间

# 解析解
def analytical_solution(t, T_s, T_0, r):
    return T_s + (T_0 - T_s) * np.exp(-r * t)

# 欧拉法数值解
def euler_method(T_s, T_0, r, delta_t, t_target):
    time_points = np.arange(0, t_target + delta_t, delta_t)
    T = np.zeros(len(time_points))
    T[0] = T_0
    for n in range(1, len(time_points)):
        T[n] = T[n - 1] + delta_t * (-r * (T[n - 1] - T_s))
    return time_points, T

# 不同时间步长
delta_ts = [0.1, 0.05, 0.025, 0.01, 0.005]
errors = []

# 计算每种步长的误差
for delta_t in delta_ts:
    _, T_numeric = euler_method(T_s, T_0, r, delta_t, t_target)
    T_exact = analytical_solution(t_target, T_s, T_0, r)
    error = abs(T_numeric[-1] - T_exact)
    errors.append(error)

# 绘制误差与步长关系
plt.figure(figsize=(10, 6))
plt.loglog(delta_ts, errors, 'o-', label='Error vs. Δt')
plt.xlabel('Time Step Δt (log scale)', fontsize=12)
plt.ylabel('Error (log scale)', fontsize=12)
plt.title('Error vs. Time Step Size in Euler Method', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# 输出结果
print("b题结果：")
for i, delta_t in enumerate(delta_ts):
    print(f"Δt = {delta_t:.3f}, Error = {errors[i]:.6f}")

# 找到合适的Δt，使误差小于0.1%
def find_delta_t(T_s, T_0, r, t_target, tolerance):
    delta_t = 0.1  # 初始步长
    while True:
        _, T_numeric = euler_method(T_s, T_0, r, delta_t, t_target)
        T_exact = analytical_solution(t_target, T_s, T_0, r)
        error = abs(T_numeric[-1] - T_exact)
        if error <= tolerance:  # 满足误差条件
            return delta_t, error
        delta_t /= 2  # 减小步长

# 对 t=1.60 和 t=5.5 进行计算
tolerance = 0.001  # 0.1% 误差
delta_t_1_60, error_1_60 = find_delta_t(T_s, T_0, r, 1.60, tolerance)
delta_t_5_5, error_5_5 = find_delta_t(T_s, T_0, r, 5.5, tolerance)

# 输出结果
print("c题结果：")
print(f"t = 1.60 分钟时，Δt = {delta_t_1_60:.6f}, Error = {error_1_60:.6f}")
print(f"t = 5.50 分钟时，Δt = {delta_t_5_5:.6f}, Error = {error_5_5:.6f}")