import numpy as np

# 常量设置
k_B = 1.38e-23  # 玻尔兹曼常数 (J/K)
Lx, Ly = 6, 6   # 系统大小
N = 18          # 粒子数
V = Lx * Ly     # 系统体积

# 示例粒子速度和力数据（需从程序输出中获取实际数据）
# 这里使用随机数生成速度和力以示例
np.random.seed(42)
m = 1.0  # 假设粒子质量为1
velocities = np.random.rand(N, 2) - 0.5  # 粒子速度 (随机模拟)
forces = np.random.rand(N, 2) - 0.5  # 粒子间力 (随机模拟)
positions = np.random.rand(N, 2) * 6  # 粒子位置 (随机模拟)

# 计算温度 T(t)
kinetic_energy = 0.5 * m * np.sum(velocities**2, axis=1)  # 每个粒子的动能
temperature = np.sum(kinetic_energy) / (N * k_B)

# 计算压力 P(t)
pairwise_distances = np.linalg.norm(positions[:, None] - positions, axis=-1)  # 粒子对间距
pairwise_forces = np.linalg.norm(forces[:, None] - forces, axis=-1)  # 粒子对间力
pressure = (N * k_B * temperature / V) + (1 / (2 * V)) * np.sum(pairwise_distances * pairwise_forces)

# 输出结果
print(f"温度 T(t): {temperature:.3e} K")
print(f"压力 P(t): {pressure:.3e} Pa")