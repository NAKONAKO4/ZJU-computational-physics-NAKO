import numpy as np

time_black = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_black = np.array([82.3, 78.5, 74.3, 70.7, 67.6, 65.0, 62.5, 60.1, 58.1, 56.1, 54.3, 52.8,
                       51.2, 49.9, 48.6, 47.2, 46.1, 45.0, 43.9, 43.0, 41.9, 41.0, 40.1, 39.5])

time_cream = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])
temp_cream = np.array([68.8, 64.8, 62.1, 59.9, 57.7, 55.9, 53.9, 52.3, 50.8, 49.5, 48.1, 46.8,
                       45.9, 44.8, 43.7, 42.6, 41.7, 40.8, 39.9, 39.3, 38.6, 37.7, 37.0, 36.4])


env_temp = 17  # 环境温度（单位：°C）

# 计算冷却常数 r 的函数
def calculate_r(time, temp, env_temp):
    r_values = []
    for i in range(len(time) - 1):
        delta_t = time[i + 1] - time[i]
        temp_diff = temp[i] - env_temp
        next_temp_diff = temp[i + 1] - env_temp
        r = -np.log(next_temp_diff / temp_diff) / delta_t
        r_values.append(r)
    return np.mean(r_values), np.std(r_values)

r_black_mean, r_black_std = calculate_r(time_black, temp_black, env_temp)
r_cream_mean, r_cream_std = calculate_r(time_cream, temp_cream, env_temp)

print("黑咖啡冷却常数 r: 平均值 = {:.4f} min^-1, 标准差 = {:.4f} min^-1".format(r_black_mean, r_black_std))
print("加奶咖啡冷却常数 r: 平均值 = {:.4f} min^-1, 标准差 = {:.4f} min^-1".format(r_cream_mean, r_cream_std))