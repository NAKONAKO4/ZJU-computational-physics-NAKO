import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann

# 参数初始化
num_particles = 16              # 粒子数
box_size_x = 10.0               # 盒子宽度 (Å)
box_size_y = 5.0                # 盒子高度 (Å)
initial_temp = 300              # 初始温度 (K)
num_steps = 10                # 模拟时间步数
dt = 0.01                        # 时间步长 (fs)
epsilon = 0.0103                # Lennard-Jones 势参数 ε (eV)
sigma = 3.4                     # Lennard-Jones 势参数 σ (Å)
mass_of_argon = 39.948          # Argon 粒子质量 (amu)


def lj_force(r, epsilon, sigma):
    """
    Lennard-Jones 势计算粒子间力.
    """
    return 48 * epsilon * np.power(
        sigma, 12) / np.power(
        r, 13) - 24 * epsilon * np.power(
        sigma, 6) / np.power(r, 7)

def lj_potential(r, epsilon, sigma):
    if r == 0:
        return 0
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def init_velocity(T, number_of_particles):
    """
    初始化粒子速度，符合指定温度 T 的 Maxwell-Boltzmann 分布.
    """
    R = np.random.rand(number_of_particles, 2) - 0.5
    return R * np.sqrt(Boltzmann * T / (mass_of_argon * 1.602e-19))


def get_accelerations(positions, box_size_x, box_size_y, epsilon, sigma):
    """
    计算每个粒子的加速度（2D），考虑周期性边界条件.
    """
    num_particles = positions.shape[0]
    accel = np.zeros_like(positions)

    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            # 计算周期性边界条件下的最小距离
            r_vec = positions[j] - positions[i]
            r_vec[0] -= box_size_x * np.round(r_vec[0] / box_size_x)
            r_vec[1] -= box_size_y * np.round(r_vec[1] / box_size_y)

            r_mag = np.linalg.norm(r_vec)  # 距离
            if r_mag > 0:
                force_scalar = lj_force(r_mag, epsilon, sigma)
                force_vec = force_scalar * r_vec / r_mag
                accel[i] += force_vec / mass_of_argon
                accel[j] -= force_vec / mass_of_argon  # 牛顿第三定律
    return accel


def update_pos(positions, velocities, accelerations, dt, box_size_x, box_size_y):
    """
    更新粒子位置，并应用周期性边界条件.
    """
    positions = positions + velocities * dt + 0.5 * accelerations * dt ** 2
    positions[:, 0] %= box_size_x  # x 方向周期性边界条件
    positions[:, 1] %= box_size_y  # y 方向周期性边界条件
    return positions


def update_velo(velocities, accelerations, new_accelerations, dt):
    """
    更新粒子速度.
    """
    return velocities + 0.5 * (accelerations + new_accelerations) * dt


def create_lattice(num_particles, box_size_x, box_size_y):
    """
    在矩形盒子内均匀放置指定数量的粒子。
    """
    positions = np.zeros((num_particles, 2))  # 初始化粒子位置数组
    num_rows = int(np.sqrt(num_particles * (box_size_y / box_size_x)))
    num_cols = int(np.ceil(num_particles / num_rows))

    x_spacing = box_size_x / num_cols  # x方向间距
    y_spacing = box_size_y / num_rows  # y方向间距

    # 使用两层循环均匀分布粒子
    index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if index >= num_particles:  # 如果粒子达到指定数量，停止添加
                break
            x_pos = (col + 0.5) * x_spacing
            y_pos = (row + 0.5) * y_spacing
            positions[index] = [x_pos, y_pos]
            index += 1

    return positions

def calculate_energy(positions, velocities, box_size_x, box_size_y, epsilon, sigma):
    """
    计算系统总能量：动能 + 势能
    """
    kinetic_energy = 0.5 * mass_of_argon * np.sum(velocities**2)  # 动能

    potential_energy = 0
    num_particles = positions.shape[0]
    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            r_vec = positions[j] - positions[i]
            r_vec[0] -= box_size_x * np.round(r_vec[0] / box_size_x)  # x方向周期性边界条件
            r_vec[1] -= box_size_y * np.round(r_vec[1] / box_size_y)  # y方向周期性边界条件
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 0:
                potential_energy += lj_potential(r_mag, epsilon, sigma)  # 势能

    total_energy = kinetic_energy + potential_energy
    return total_energy

def calculate_momentum(velocities):
    """
    计算系统总动量
    """
    momentum = mass_of_argon * np.sum(velocities, axis=0)  # 动量向量
    return momentum

def run_md(num_particles, box_size_x, box_size_y, initial_temp, num_steps, dt, epsilon, sigma):
    """
    运行 2D 分子动力学模拟.
    """
    positions = create_lattice(num_particles, box_size_x, box_size_y)
    velocities = init_velocity(initial_temp, num_particles)
    print(len(positions), len(velocities))
    accelerations = get_accelerations(positions, box_size_x, box_size_y, epsilon, sigma)

    sim_positions = np.zeros((num_steps, num_particles, 2))
    energies = []
    momenta = []

    for step in range(num_steps):
        positions = update_pos(positions, velocities, accelerations, dt, box_size_x, box_size_y)
        new_accelerations = get_accelerations(positions, box_size_x, box_size_y, epsilon, sigma)
        velocities = update_velo(velocities, accelerations, new_accelerations, dt)
        accelerations = new_accelerations
        sim_positions[step] = positions
        total_energy = calculate_energy(positions, velocities, box_size_x, box_size_y, epsilon, sigma)
        momentum = calculate_momentum(velocities)
        energies.append(total_energy)
        momenta.append(np.linalg.norm(momentum))

    return sim_positions, energies, momenta


# 运行模拟
sim_positions, energies, momenta = run_md(num_particles, box_size_x, box_size_y, initial_temp, num_steps, dt, epsilon, sigma)

# 可视化结果
for i in range(sim_positions.shape[1]):
    plt.plot(sim_positions[:, i, 0], sim_positions[:, i, 1], '.', label=f'atom {i}')
plt.xlabel('X-Position (Å)')
plt.ylabel('Y-Position (Å)')
plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 绘制能量随时间的变化
plt.figure()
plt.plot(range(num_steps), energies, label='Total Energy')
plt.xlabel('Time Step')
plt.ylabel('Energy (eV)')
plt.title('Total Energy Evolution')
plt.legend()
plt.show()

# 绘制动量随时间的变化
plt.figure()
plt.plot(range(num_steps), momenta, label='Total Momentum')
plt.xlabel('Time Step')
plt.ylabel('Momentum (amu·Å/fs)')
plt.title('Total Momentum Evolution')
plt.legend()
plt.show()