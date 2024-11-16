import numpy as np

# 冷却常数计算函数
def calculate_r(time, temp, T_env):
    r_values = []
    for i in range(len(time) - 1):
        delta_t = time[i + 1] - time[i]
        temp_diff = temp[i] - T_env
        next_temp_diff = temp[i + 1] - T_env
        r = -np.log(next_temp_diff / temp_diff) / delta_t
        r_values.append(r)
    return np.mean(r_values)

# 原始数据
time_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

time_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                       45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])

T_env = 17  # 环境温度

# 不同时间步长
# 黑咖啡
time_black_large_step = time_black[::2]  # 增大步长，每隔 4 分钟
temp_black_large_step = temp_black[::2]

time_black_small_step = np.linspace(0, 46, 47)  # 减小步长，每 1 分钟
temp_black_small_step = np.interp(time_black_small_step, time_black, temp_black)  # 插值温度

# 奶咖啡
time_cream_large_step = time_cream[::2]  # 增大步长，每隔 4 分钟
temp_cream_large_step = temp_cream[::2]

time_cream_small_step = np.linspace(0, 46, 47)  # 减小步长，每 1 分钟
temp_cream_small_step = np.interp(time_cream_small_step, time_cream, temp_cream)  # 插值温度

# 计算冷却常数 r
# 黑咖啡
r_black_original = calculate_r(time_black, temp_black, T_env)  # 原始时间步长
r_black_large_step = calculate_r(time_black_large_step, temp_black_large_step, T_env)  # 较大时间步长
r_black_small_step = calculate_r(time_black_small_step, temp_black_small_step, T_env)  # 较小时间步长

# 奶咖啡
r_cream_original = calculate_r(time_cream, temp_cream, T_env)  # 原始时间步长
r_cream_large_step = calculate_r(time_cream_large_step, temp_cream_large_step, T_env)  # 较大时间步长
r_cream_small_step = calculate_r(time_cream_small_step, temp_cream_small_step, T_env)  # 较小时间步长

# 计算误差
# 黑咖啡误差
relative_error_black_large = abs(r_black_large_step - r_black_original) / r_black_original * 100
relative_error_black_small = abs(r_black_small_step - r_black_original) / r_black_original * 100

# 奶咖啡误差
relative_error_cream_large = abs(r_cream_large_step - r_cream_original) / r_cream_original * 100
relative_error_cream_small = abs(r_cream_small_step - r_cream_original) / r_cream_original * 100

# 输出结果
print("黑咖啡冷却常数 r:")
print(f"  原始步长: r = {r_black_original:.4f}")
print(f"  较大步长: r = {r_black_large_step:.4f} (相对误差: {relative_error_black_large:.2f}%)")
print(f"  较小步长: r = {r_black_small_step:.4f} (相对误差: {relative_error_black_small:.2f}%)")

print("\n奶咖啡冷却常数 r:")
print(f"  原始步长: r = {r_cream_original:.4f}")
print(f"  较大步长: r = {r_cream_large_step:.4f} (相对误差: {relative_error_cream_large:.2f}%)")
print(f"  较小步长: r = {r_cream_small_step:.4f} (相对误差: {relative_error_cream_small:.2f}%)")