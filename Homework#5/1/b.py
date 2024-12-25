import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------- 创建输出文件夹 -------------------------
output_folder = 'plots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# --------------------------- 辅助函数定义 --------------------------

def initialize(N, Lx, Ly):
    """
    初始化粒子的位置和速度。
    位置在 [0, Lx] × [0, Ly] 内随机分布。
    速度在 [-1, 1] 之间随机初始化，并调整为零总动量。
    """
    # 随机初始化位置
    x = Lx * np.random.rand(2, N)

    # 随机初始化速度
    v = (np.random.rand(2, N) - 0.5) * 2  # 速度范围 [-1, 1]

    # 调整速度以确保总动量为零
    v_avg = np.mean(v, axis=1, keepdims=True)
    v -= v_avg

    return x, v


def force(x, Lx, Ly):
    """
    计算粒子间的作用力。
    使用平方反比力模型 F = r / |r|^3 = 1 / r^2 * unit_vector
    返回力矩阵 F 和 virial_sum。
    """
    N = x.shape[1]
    F = np.zeros((2, N))
    virial_sum = 0.0

    for i in range(N - 1):
        for j in range(i + 1, N):
            # 计算两粒子间的位移向量，考虑周期性边界条件
            dx = x[:, j] - x[:, i]
            dx[0] -= Lx * np.round(dx[0] / Lx)
            dx[1] -= Ly * np.round(dx[1] / Ly)

            r2 = np.dot(dx, dx)
            if r2 < 1e-12:
                continue  # 避免除以零

            r = np.sqrt(r2)
            F_pair = dx / r ** 3  # 力的大小和方向

            # 累加力
            F[:, i] += F_pair
            F[:, j] -= F_pair

            # 累加 virial
            virial_sum += np.dot(dx, F_pair)

    return F, virial_sum


def count_particles_left_half(x, Lx):
    """
    计算左半部分盒子中的粒子数。
    """
    x_coords = x[0, :]
    return np.sum(x_coords < (Lx / 2))


def run_simulation(N, Lx_initial, Ly_initial, dt, t_max, change_box, new_Lx, new_Ly):
    """
    运行分子动力学模拟。

    参数:
    - N: 粒子数
    - Lx_initial, Ly_initial: 初始盒子尺寸
    - dt: 时间步长
    - t_max: 最大模拟时间
    - change_box: 是否改变盒子尺寸
    - new_Lx, new_Ly: 新的盒子尺寸（如果 change_box 为 True）

    返回:
    - t: 时间向量
    - T: 温度随时间变化
    - P_calculated: 计算得到的压力
    - P_ideal: 理想气体压力
    - pressure_difference: 压力差值
    - KE: 总动能随时间变化
    - momentum: 总动量随时间变化
    - n_t: 左半部分盒子中的粒子数随时间变化
    - n_avg: 粒子数的时间平均值随时间变化
    - x: 粒子位置随时间变化
    """
    num_steps = int(t_max / dt) + 1
    t = np.linspace(0, t_max, num_steps)

    # 初始化盒子尺寸
    Lx = Lx_initial
    Ly = Ly_initial

    # 初始化粒子位置和速度
    x, v = initialize(N, Lx, Ly)

    # 初始化加速度
    F, virial_sum = force(x, Lx, Ly)
    a = F  # 假设质量 m = 1

    # 初始化存储变量
    T = np.zeros(num_steps)
    P_calculated = np.zeros(num_steps)
    P_ideal = np.zeros(num_steps)
    pressure_difference = np.zeros(num_steps)
    KE = np.zeros(num_steps)
    momentum = np.zeros((2, num_steps))
    n_t = np.zeros(num_steps)
    n_avg = np.zeros(num_steps)

    # 玻尔兹曼常数
    kB = 1.0  # 与 MATLAB 代码保持一致

    # 计算初始动能和动量
    KE[0] = 0.5 * np.sum(v ** 2)
    momentum[:, 0] = np.sum(v, axis=1)
    T[0] = (2 * KE[0]) / (2 * N * kB)  # 2D系统，自由度是 2N
    P_calculated[0] = (N * kB * T[0] + 0.5 * virial_sum) / (Lx * Ly)
    P_ideal[0] = (N * kB * T[0]) / (Lx * Ly)
    pressure_difference[0] = P_calculated[0] - P_ideal[0]
    n_t[0] = count_particles_left_half(x, Lx)
    n_avg[0] = n_t[0]

    # 主循环
    for i in range(1, num_steps):
        # 在特定时间步改变盒子尺寸
        if change_box and i == 1:
            Lx = new_Lx
            Ly = new_Ly
            # 调整粒子位置以适应新的盒子尺寸，逐维度处理
            x[0, :] = np.mod(x[0, :], Lx)
            x[1, :] = np.mod(x[1, :], Ly)

        # Velocity Verlet 步骤
        # 1. 更新位置
        x += v * dt + 0.5 * a * dt ** 2

        # 2. 应用周期性边界条件，逐维度处理
        x[0, :] = np.mod(x[0, :], Lx)
        x[1, :] = np.mod(x[1, :], Ly)

        # 3. 计算新的力和 virial
        F_new, virial_sum_new = force(x, Lx, Ly)

        # 4. 更新速度
        v += 0.5 * (a + F_new) * dt  # 假设质量 m = 1

        # 5. 更新加速度
        a = F_new

        # 6. 计算动能和动量
        KE[i] = 0.5 * np.sum(v ** 2)
        momentum[:, i] = np.sum(v, axis=1)

        # 7. 计算温度
        T[i] = (2 * KE[i]) / (2 * N * kB)  # 2D系统，自由度是 2N

        # 8. 计算压力
        P_calculated[i] = (N * kB * T[i] + 0.5 * virial_sum_new) / (Lx * Ly)

        # 9. 计算理想气体压强
        P_ideal[i] = (N * kB * T[i]) / (Lx * Ly)

        # 10. 计算压力差值
        pressure_difference[i] = P_calculated[i] - P_ideal[i]

        # 11. 计算左半部分盒子中的粒子数
        n_t[i] = count_particles_left_half(x, Lx)

        # 12. 计算粒子数的时间平均值
        if i == 0:
            n_avg[i] = n_t[i]
        else:
            n_avg[i] = np.mean(n_t[:i + 1])

    return t, T, P_calculated, P_ideal, pressure_difference, KE, momentum, n_t, n_avg, x


def plot_temperature_pressure(t, T, P_calculated, P_ideal, pressure_diff, part_label):
    """
    绘制温度、压力和压力差值随时间变化的图形，并保存为 PNG 文件。
    """
    plt.figure(figsize=(20, 12))  # 宽度2000像素，高度1200像素

    # 温度随时间变化
    plt.subplot(3, 1, 1)
    plt.plot(t, T, linewidth=2)
    plt.title(f'({part_label}) 温度随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('温度 T(t)', fontsize=16)
    plt.tick_params(labelsize=14)

    # 压强随时间变化
    plt.subplot(3, 1, 2)
    plt.plot(t, P_calculated, linewidth=2, label='计算得到的压强 P(t)')
    plt.plot(t, P_ideal, '--', linewidth=2, label='理想气体压强 P_{ideal}(t)')
    plt.title(f'({part_label}) 压强随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('压强', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(labelsize=14)

    # 压强差值
    plt.subplot(3, 1, 3)
    plt.plot(t, pressure_diff, linewidth=2)
    plt.title(f'({part_label}) 压强差值 ΔP(t) = P(t) - P_ideal(t)', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('压强差值 ΔP(t)', fontsize=16)
    plt.tick_params(labelsize=14)

    plt.suptitle(f'部分 ({part_label}) 结果', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整标题空间

    # 保存图形
    plt.savefig(os.path.join(output_folder, f'part_{part_label.lower()}_temperature_pressure.png'), dpi=300)
    plt.close()


def plot_energy_momentum(t, KE, momentum, part_label):
    """
    绘制总动能和总动量随时间变化的图形，并保存为 PNG 文件。
    """
    plt.figure(figsize=(20, 12))  # 宽度2000像素，高度1200像素

    # 总动能随时间变化
    plt.subplot(2, 1, 1)
    plt.plot(t, KE, linewidth=2)
    plt.title(f'({part_label}) 总动能随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('总动能', fontsize=16)
    plt.tick_params(labelsize=14)

    # 总动量随时间变化
    plt.subplot(2, 1, 2)
    plt.plot(t, momentum[0, :], linewidth=2, label='动量 x 分量')
    plt.plot(t, momentum[1, :], linewidth=2, label='动量 y 分量')
    plt.title(f'({part_label}) 总动量随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('总动量', fontsize=16)
    plt.legend(fontsize=14, loc='upper right')
    plt.tick_params(labelsize=14)

    plt.suptitle(f'部分 ({part_label}) 总能量和总动量守恒性检查', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整标题空间

    # 保存图形
    plt.savefig(os.path.join(output_folder, f'part_{part_label.lower()}_energy_momentum.png'), dpi=300)
    plt.close()


def plot_particle_number(t, n_t, n_avg, part_label):
    """
    绘制左半部分盒子中的粒子数及其时间平均值随时间变化的图形，并保存为 PNG 文件。
    """
    plt.figure(figsize=(20, 12))  # 宽度2000像素，高度1200像素

    # 粒子数 n(t) 随时间变化
    plt.subplot(2, 1, 1)
    plt.plot(t, n_t, linewidth=2)
    plt.title(f'({part_label}) 左半部分盒子中的粒子数 n(t) 随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('粒子数 n(t)', fontsize=16)
    plt.tick_params(labelsize=14)

    # 粒子数的时间平均值随时间变化
    plt.subplot(2, 1, 2)
    plt.plot(t, n_avg, linewidth=2)
    plt.title(f'({part_label}) 粒子数 n(t) 的时间平均值随时间变化', fontsize=18)
    plt.xlabel('时间', fontsize=16)
    plt.ylabel('平均粒子数 ⟨n(t)⟩', fontsize=16)
    plt.tick_params(labelsize=14)

    plt.suptitle(f'部分 ({part_label}) 结果', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整标题空间

    # 保存图形
    plt.savefig(os.path.join(output_folder, f'part_{part_label.lower()}_particle_number.png'), dpi=300)
    plt.close()


# --------------------------- 主脚本部分 ---------------------------

def main():
    # 部分 (a)
    print('开始部分 (a): 系统 Lx × Ly = 6 × 6, N = 18, Δt = 0.01')
    N_a = 18
    Lx_a = 6
    Ly_a = 6
    dt_a = 0.01
    t_max_a = 10
    change_box_a = False
    new_Lx_a = Lx_a
    new_Ly_a = Ly_a

    t_a, T_a, P_calculated_a, P_ideal_a, pressure_diff_a, KE_a, momentum_a, n_t_a, n_avg_a, x_a = run_simulation(
        N_a, Lx_a, Ly_a, dt_a, t_max_a, change_box_a, new_Lx_a, new_Ly_a
    )

    # 绘制温度和压力
    plot_temperature_pressure(t_a, T_a, P_calculated_a, P_ideal_a, pressure_diff_a, 'a')

    # 绘制总动能和总动量
    plot_energy_momentum(t_a, KE_a, momentum_a, 'a')

    # 部分 (b)
    print('开始部分 (b): 在 t = 0 时将盒子尺寸从 6 × 6 改变为 12 × 6')
    N_b = 18
    Lx_initial_b = 6
    Ly_initial_b = 6
    dt_b = 0.01
    t_max_b = 10
    change_box_b = True
    new_Lx_b = 12
    new_Ly_b = 6

    t_b, T_b, P_calculated_b, P_ideal_b, pressure_diff_b, KE_b, momentum_b, n_t_b, n_avg_b, x_b = run_simulation(
        N_b, Lx_initial_b, Ly_initial_b, dt_b, t_max_b, change_box_b, new_Lx_b, new_Ly_b
    )

    # 绘制温度和压力
    plot_temperature_pressure(t_b, T_b, P_calculated_b, P_ideal_b, pressure_diff_b, 'b')

    # 绘制总动能和总动量
    plot_energy_momentum(t_b, KE_b, momentum_b, 'b')

    # 部分 (c)
    print('开始部分 (c): 计算左半部分盒子中的粒子数 n(t) 及其时间平均值')
    N_c = 18
    Lx_c = 6
    Ly_c = 6
    dt_c = 0.01
    t_max_c = 10
    change_box_c = False
    new_Lx_c = Lx_c
    new_Ly_c = Ly_c

    t_c, T_c, P_calculated_c, P_ideal_c, pressure_diff_c, KE_c, momentum_c, n_t_c, n_avg_c, x_c = run_simulation(
        N_c, Lx_c, Ly_c, dt_c, t_max_c, change_box_c, new_Lx_c, new_Ly_c
    )

    # 绘制粒子数
    plot_particle_number(t_c, n_t_c, n_avg_c, 'c')

    # 描述n(t)的定性行为
    print(f'部分 (c) 的时间平均粒子数: {n_avg_c[-1]} (N = {N_c})')

    # 部分 (d)
    print('开始部分 (d): 系统 Lx × Ly = 6 × 6, N = 36，重复部分 (a) 的计算')
    N_d = 36
    Lx_d = 6
    Ly_d = 6
    dt_d = 0.01
    t_max_d = 10
    change_box_d = False
    new_Lx_d = Lx_d
    new_Ly_d = Ly_d

    t_d, T_d, P_calculated_d, P_ideal_d, pressure_diff_d, KE_d, momentum_d, n_t_d, n_avg_d, x_d = run_simulation(
        N_d, Lx_d, Ly_d, dt_d, t_max_d, change_box_d, new_Lx_d, new_Ly_d
    )

    # 绘制温度和压力
    plot_temperature_pressure(t_d, T_d, P_calculated_d, P_ideal_d, pressure_diff_d, 'd')

    # 绘制总动能和总动量
    plot_energy_momentum(t_d, KE_d, momentum_d, 'd')

    # 部分 (e)
    print('开始部分 (e): 系统 Lx × Ly = 12 × 12, N = 72，重复部分 (a) 的计算')
    N_e = 72
    Lx_e = 12
    Ly_e = 12
    dt_e = 0.01
    t_max_e = 10
    change_box_e = False
    new_Lx_e = Lx_e
    new_Ly_e = Ly_e

    t_e, T_e, P_calculated_e, P_ideal_e, pressure_diff_e, KE_e, momentum_e, n_t_e, n_avg_e, x_e = run_simulation(
        N_e, Lx_e, Ly_e, dt_e, t_max_e, change_box_e, new_Lx_e, new_Ly_e
    )

    # 绘制温度和压力
    plot_temperature_pressure(t_e, T_e, P_calculated_e, P_ideal_e, pressure_diff_e, 'e')

    # 绘制总动能和总动量
    plot_energy_momentum(t_e, KE_e, momentum_e, 'e')

    print('所有部分已完成。请查看 "plots" 文件夹中的 PNG 图片以分析结果。')


# --------------------------- 执行主脚本 ---------------------------
if __name__ == "__main__":
    main()