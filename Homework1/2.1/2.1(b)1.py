import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 冷却模型定义
def cooling_model(t, r, T_env, T_initial):
    return T_env + (T_initial - T_env) * np.exp(-r * t)

# 数据输入
time_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])
time_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                       45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

# 环境温度
ambient_temp = 17
# 初始猜测值
r_guess = 0.03  # 猜测的冷却常数
initial_temp_black = temp_black[0]  # 黑咖啡初始温度
initial_temp_cream = temp_cream[0]  # 奶咖啡初始温度

# 黑咖啡拟合
def exponential_model(t, r, T0):
    return ambient_temp + (T0 - ambient_temp) * np.exp(-r * t)

# 初始猜测值
initial_guess = [0.0, 82.3]  # r 和初始温度 T0
params_black, _ = curve_fit(exponential_model, time_black, temp_black, p0=initial_guess)
r_black = params_black[0]

# 奶咖啡拟合
initial_guess = [0.0, 68.8]  # r 和初始温度 T0
params_cream, _ = curve_fit(exponential_model, time_cream, temp_cream, p0=initial_guess)
r_cream = params_cream[0]

# 使用拟合的 r 计算温度
time_fit = np.linspace(0, 46, 200)  # 生成更多时间点用于绘图
temp_fit_black = cooling_model(time_fit, r_black, ambient_temp, initial_temp_black)
temp_fit_cream = cooling_model(time_fit, r_cream, ambient_temp, initial_temp_cream)

# 绘制实验数据和拟合曲线
plt.figure(figsize=(12, 8))

# 黑咖啡
plt.plot(time_black, temp_black, 'o', label='Experimental Data (Black Coffee)')
plt.plot(time_fit, temp_fit_black, '-', label=f'Fitted Curve (Black Coffee, r = {r_black:.4f})')

# 奶咖啡
plt.plot(time_cream, temp_cream, 's', label='Experimental Data (Cream Coffee)')
plt.plot(time_fit, temp_fit_cream, '--', label=f'Fitted Curve (Cream Coffee, r = {r_cream:.4f})')

# 图表设置
plt.xlabel('Time (minutes)', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.title('Cooling of Black Coffee and Cream Coffee with Initial Guess', fontsize=16)
plt.legend()
plt.grid()
plt.show()