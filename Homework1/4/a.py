import numpy as np
import matplotlib.pyplot as plt

# 参数定义
g = 9.8  # 重力加速度 (m/s^2)
v0 = 25  # 初始速度 (m/s)
angles = np.radians([15, 30, 45, 60, 75])  # 不同的发射角 (度转弧度)

# 抛射轨迹计算函数
def trajectory(v0, theta, g):
    t_flight = 2 * v0 * np.sin(theta) / g  # 飞行时间
    t = np.linspace(0, t_flight, num=500)  # 时间点
    x = v0 * np.cos(theta) * t  # 水平位移
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2  # 垂直位移
    return x, y

# 绘制轨迹
plt.figure(figsize=(10, 6))
for theta in angles:
    x, y = trajectory(v0, theta, g)
    plt.plot(x, y, label=f"θ = {np.degrees(theta):.0f}°")

plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.title("Trajectory of a Ball (No Air Resistance)")
plt.legend()
plt.grid()
plt.show()

# 从不同高度抛出
def range_from_height(v0, theta, h, g):
    a = -0.5 * g
    b = v0 * np.sin(theta)
    c = h
    t_up = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # 解二次方程找时间
    t_down = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    t_flight = max(t_up, t_down)
    R = v0 * np.cos(theta) * t_flight
    return R

# 测试角度
angles = np.radians(np.linspace(30, 60, 100))
ranges = [range_from_height(v0, theta, 2.1, g) for theta in angles]

# 找到最大射程对应的角度
max_range_idx = np.argmax(ranges)
optimal_angle = np.degrees(angles[max_range_idx])
max_range = ranges[max_range_idx]

print(f"最大射程: {max_range:.2f} m, 最佳角度: {optimal_angle:.2f}°")

# 绘制射程 vs 角度
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(angles), ranges, label="Range vs Angle")
plt.axvline(optimal_angle, color="r", linestyle="--", label=f"Optimal Angle: {optimal_angle:.2f}°")
plt.xlabel("Angle (degrees)")
plt.ylabel("Range (m)")
plt.title("Range vs Angle for h = 2.1 m")
plt.legend()
plt.grid()
plt.show()


# 带空气阻力的抛射运动
def trajectory_with_drag(v0, theta, h, g, C, m, dt=0.01):
    k2 = C / m
    t = [0]
    x, y = [0], [h]
    vx, vy = [v0 * np.cos(theta)], [v0 * np.sin(theta)]

    while y[-1] >= 0:
        v = np.sqrt(vx[-1] ** 2 + vy[-1] ** 2)
        ax = -k2 * v * vx[-1]
        ay = -g - k2 * v * vy[-1]

        # 更新速度和位置
        vx.append(vx[-1] + ax * dt)
        vy.append(vy[-1] + ay * dt)
        x.append(x[-1] + vx[-1] * dt)
        y.append(y[-1] + vy[-1] * dt)
        t.append(t[-1] + dt)

    return np.array(x), np.array(y)


# 模拟轨迹
theta = np.radians(45)
C = 0.1
m = 1.0
x_drag, y_drag = trajectory_with_drag(v0, theta, 2.1, g, C, m)

# 绘制比较
x_no_drag, y_no_drag = trajectory(v0, theta, g)

plt.figure(figsize=(10, 6))
plt.plot(x_no_drag, y_no_drag, label="No Air Resistance", linestyle="--")
plt.plot(x_drag, y_drag, label="With Air Resistance")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.title("Trajectory with and without Air Resistance")
plt.legend()
plt.grid()
plt.show()