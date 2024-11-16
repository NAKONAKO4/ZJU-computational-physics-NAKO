import numpy as np
from scipy.optimize import curve_fit

# 实验数据
time = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
ambient_temp = 17  # 环境温度

# 方式 1：逐段计算冷却常数
def calculate_r(time, temp, ambient_temp):
    r_values = []
    for i in range(len(time) - 1):
        delta_t = time[i + 1] - time[i]
        temp_diff = temp[i] - ambient_temp
        next_temp_diff = temp[i + 1] - ambient_temp
        r = -np.log(next_temp_diff / temp_diff) / delta_t
        r_values.append(r)
    return np.mean(r_values), np.std(r_values)

r_black_mean, r_black_std = calculate_r(time, temp_black, ambient_temp)

# 方式 2：整体拟合指数模型
def exponential_model(t, r, T0):
    return ambient_temp + (T0 - ambient_temp) * np.exp(-r * t)

# 初始猜测值
initial_guess = [0.0, 82.3]  # r 和初始温度 T0
params, _ = curve_fit(exponential_model, time, temp_black, p0=initial_guess)
r_black_fit = params[0]

# 输出结果
print("方式 1：逐段计算")
print("平均冷却常数 r = {:.4f}, 标准差 = {:.4f}".format(r_black_mean, r_black_std))

print("\n方式 2：整体拟合")
print("拟合冷却常数 r = {:.4f}".format(r_black_fit))

# 比较结果
if abs(r_black_fit - r_black_mean) < r_black_std:
    print("\n整体拟合与逐段计算结果相符，但整体拟合更能平滑波动，适合描述整体趋势。")
else:
    print("\n两种方法结果不同，逐段计算更适合描述局部变化，整体拟合适合描述整体趋势。")