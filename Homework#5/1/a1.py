import numpy as np
import matplotlib.pyplot as plt

# 常数定义

mass = 1  # 单个粒子质量（单位：千克）
k_B = 1.38e-23  # 玻尔兹曼常数（单位：焦耳/开尔文）
Lx, Ly = 12.0, 6.0  # 系统的尺寸（单位：米）
temperature = 100  # 初始温度（单位：开尔文）
N = 18  # 粒子数
dt = 0.005  # 时间步长
dt2 = dt ** 2  # 时间步长的平方
area = Lx * Ly  # 体系的面积（二维）


# 粒子的位置初始化
def create_lattice(N, Lx, Ly):
    positions = np.zeros((N, 2))
    num_rows = int(np.sqrt(N))
    num_cols = int(np.ceil(N / num_rows))
    x_spacing = Lx / num_cols
    y_spacing = Ly / num_rows

    index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if index >= N:
                break
            x_pos = (col + 0.5) * x_spacing
            y_pos = (row + 0.5) * y_spacing
            positions[index] = [x_pos, y_pos]
            index += 1
    return positions


# 粒子的速度初始化
def init_velocity(N, temperature, mass):
    stddev = np.sqrt(k_B * temperature / mass)
    velocities = np.random.normal(0, stddev, (N, 2))
    velocities -= np.mean(velocities, axis=0)  # 保证总动量为0
    return velocities


# 周期性边界条件
def pbc(pos, L):
    return (pos + L) % L


# 计算粒子间的力和势能
def force(dx, dy):
    r2 = dx ** 2 + dy ** 2
    if r2 == 0:  # 避免除以零
        return 0, 0, 0
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    f = 24 * r6_inv * (2 * r6_inv - 1) * r2_inv
    fx = f * dx
    fy = f * dy
    potential = 4 * (r6_inv - r6_inv ** 2)
    return fx, fy, potential


# 计算加速度、势能和维里尔
def accel(x, y):
    ax = np.zeros(N)
    ay = np.zeros(N)
    pe = 0.0
    virial = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = pbc(x[i] - x[j], Lx)
            dy = pbc(y[i] - y[j], Ly)
            fx, fy, pot = force(dx, dy)
            ax[i] += fx
            ay[i] += fy
            ax[j] -= fx
            ay[j] -= fy
            pe += pot
            virial += dx * fx + dy * fy
    return ax, ay, pe, virial


# Verlet积分更新粒子的位置和速度
def verlet(x, y, vx, vy, ax, ay):
    # 更新位置
    x += vx * dt + 0.5 * ax * dt2
    y += vy * dt + 0.5 * ay * dt2
    # 周期性边界条件
    x = pbc(x, Lx)
    y = pbc(y, Ly)

    # 计算新的加速度、势能和维里尔
    ax_new, ay_new, pe, virial = accel(x, y)

    # 更新速度
    vx += 0.5 * (ax + ax_new) * dt
    vy += 0.5 * (ay + ay_new) * dt
    ke = 0.5 * mass * np.sum(vx ** 2 + vy ** 2)  # 计算动能
    return x, y, vx, vy, ax_new, ay_new, ke, pe, virial


# 计算总动量
def compute_momentum(vx, vy):
    px = np.sum(mass * vx)
    py = np.sum(mass * vy)
    return np.sqrt(px ** 2 + py ** 2)


# 初始化
positions = create_lattice(N, Lx, Ly)
velocities = init_velocity(N, temperature, mass)
x, y = positions[:, 0], positions[:, 1]
vx, vy = velocities[:, 0], velocities[:, 1]
ax = np.zeros(N)
ay = np.zeros(N)

# 存储数据
t_values = []
T_values = []
E_values = []
P_values = []
momentum_values = []
pressure_diff_values = []

# 主循环
t = 0.0
while t < 20.0:
    # 使用Verlet积分更新粒子位置和速度
    x, y, vx, vy, ax, ay, ke, pe, virial = verlet(x, y, vx, vy, ax, ay)

    total_energy = ke + pe
    total_momentum = compute_momentum(vx, vy)

    # 计算压力 (Virial方法)
    pressure = (2 / 3 * ke / area) + (virial / (2 * area))

    # 计算温度（使用动能和理想气体公式）
    T = (2 * ke) / (3 * N * k_B)

    # 计算理想气体压强
    P_ideal = (N * k_B * T) / area

    # 计算压强差值
    pressure_diff = pressure - P_ideal

    # 存储数据
    t_values.append(t)
    E_values.append(total_energy)
    P_values.append(pressure)
    momentum_values.append(total_momentum)
    pressure_diff_values.append(pressure_diff)
    T_values.append(T)

    t += dt  # 更新时间

# 绘制结果
plt.figure(figsize=(12, 8))

# 总能量
plt.subplot(2, 1, 1)
plt.plot(t_values, E_values, label='Total Energy (E)', color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Total Energy (E)')
plt.title('Total Energy vs Time')
plt.grid(True)
plt.legend()

# 总动量
plt.subplot(2, 1, 2)
plt.plot(t_values, momentum_values, label='Total Momentum', color='red')
plt.xlabel('Time (t)')
plt.ylabel('Total Momentum')
plt.title('Total Momentum vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 温度和压力图
plt.figure(figsize=(12, 8))

# 温度
plt.subplot(2, 1, 1)
plt.plot(t_values, T_values, label='Temperature', color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.grid(True)
plt.legend()

# 压力
plt.subplot(2, 1, 2)
plt.plot(t_values, P_values, label='Pressure', color='red')
plt.xlabel('Time (t)')
plt.ylabel('Pressure')
plt.title('Pressure vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 压力差异图
plt.figure(figsize=(10, 6))
plt.plot(t_values, pressure_diff_values, label="Pressure Difference (Simulation - Ideal)")
plt.xlabel('Time (s)')
plt.ylabel('Pressure Difference (Pa)')
plt.title('Difference Between Simulation Pressure and Ideal Gas Pressure')
plt.legend()
plt.grid(True)
plt.show()