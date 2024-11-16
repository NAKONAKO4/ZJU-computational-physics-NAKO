import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 牛顿冷却定律函数
def cooling_model(t, r, T_env, T_initial):
    return T_env + (T_initial - T_env) * np.exp(-r * t)

# 实验数据
time_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temperature_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

# 环境温度和初始温度
T_env = 17
T_initial = temperature_black[0]

# 使用最小二乘法拟合来计算 r 值
params, _ = curve_fit(lambda t, r: cooling_model(t, r, T_env, T_initial), time_black, temperature_black)
r = params[0]

# 打印拟合的 r 值
print(f"拟合的冷却常数 r 值为: {r:.5f} min⁻¹")

# 使用拟合的 r 值计算模型温度并绘制结果
time_fit = np.linspace(0, 46, 100)
temperature_fit = cooling_model(time_fit, r, T_env, T_initial)

plt.plot(time_black, temperature_black, 'o', label='实验数据 (黑咖啡)')
plt.plot(time_fit, temperature_fit, '-', label=f'拟合曲线 (r={r:.5f})')
plt.xlabel('时间 (min)')
plt.ylabel('温度 (°C)')
plt.legend()
plt.title('咖啡冷却过程')
plt.show()