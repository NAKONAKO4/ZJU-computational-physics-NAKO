import numpy as np
import matplotlib.pyplot as plt

sigma = 0.1
epsilon = 1
mass = 6.69e-26
tau = sigma * np.sqrt(mass / epsilon)
k_B = 1.38e-23

dt = 0.005
dt2 = dt**2
N = 18
Lx, Ly = 6.0, 6.0
k_B = 1.38e-23
temperature = 100
m = 1
area = Lx * Ly

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

def init_velocity(N, temperature, mass):
    stddev = np.sqrt(k_B * temperature / mass)
    velocities = np.random.normal(0, stddev, (N, 2))
    velocities -= np.mean(velocities, axis=0)
    return velocities

positions = create_lattice(N, Lx, Ly)
velocities = init_velocity(N, temperature, m)

x, y = positions[:, 0], positions[:, 1]
vx, vy = velocities[:, 0], velocities[:, 1]
print(vx)
ax = np.zeros(N)
ay = np.zeros(N)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, c='blue', label='Particles', s=100)
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X Position (Å)')
plt.ylabel('Y Position (Å)')
plt.title('Particle Positions in 6x6 Box')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()


def check_momentum(vx, vy):
    vx -= np.sum(vx) / N
    vy -= np.sum(vy) / N
    return vx, vy

def pbc(pos, L):
    return (pos + L) % L

def separation(ds, L):
    return ds - L * np.round(ds / L)

def force(dx, dy):
    r2 = dx**2 + dy**2
    if r2 == 0:
        return 0, 0, 0
    rm2 = 1.0 / r2
    rm6 = rm2**3
    f_over_r = 24 * rm6 * (2 * rm6 - 1) * rm2
    fx = f_over_r * dx
    fy = f_over_r * dy
    pot = 4.0 * (rm6**2 - rm6)
    return fx, fy, pot

def accel(x, y, ax, ay):
    ax.fill(0)
    ay.fill(0)
    pe = 0.0
    virial = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            dx = separation(x[i] - x[j], Lx)
            dy = separation(y[i] - y[j], Ly)
            fx, fy, pot = force(dx, dy)
            ax[i] += fx
            ay[i] += fy
            ax[j] -= fx
            ay[j] -= fy
            pe += pot
            virial += dx* fx + dy* fy
    return pe, virial

def verlet(x, y, vx, vy, ax, ay):
    x += vx * dt + 0.5 * ax * dt2
    y += vy * dt + 0.5 * ay * dt2
    x = pbc(x, Lx)
    y = pbc(y, Ly)
    vx += 0.5 * ax * dt
    vy += 0.5 * ay * dt
    pe, virial = accel(x, y, ax, ay)
    vx += 0.5 * ax * dt
    vy += 0.5 * ay * dt
    ke = 0.5 * m * np.sum(vx**2 + vy**2)
    return ke, pe, virial

def compute_momentum(vx, vy):
    px = np.sum(m * vx)
    py = np.sum(m * vy)
    total_momentum = np.sqrt(px**2 + py**2)
    return total_momentum

vx, vy = check_momentum(vx, vy)
ke = 0.5 * np.sum(vx**2 + vy**2)
pe, virial = accel(x, y, ax, ay)
total_momentum = compute_momentum(vx, vy)
print(f"{'time':>6} {'E':>12} {'Momentum':>12} {'T':>12} {'P':>12}")

t_values = []
T_values = []
E_values = []
P_values = []
momentum_values = []
pressure_diff_values = []

t = 0.0
while t < 5.0:
    d=2
    ke, pe, virial = verlet(x, y, vx, vy, ax, ay)
    total_energy = ke + pe
    total_momentum = compute_momentum(vx, vy)
    pressure = (0.5 * virial) / area

    T = (2 * ke*epsilon) / ((d * N-d) * k_B)
    pressure = (N * k_B * T / (area*(sigma**2))) + (virial / (d * area))
    P_ideal = (N * k_B *T) / (area*(sigma**2))

    pressure_diff = pressure - P_ideal

    t_values.append(t)
    E_values.append(total_energy)
    P_values.append(pressure)
    momentum_values.append(total_momentum)
    pressure_diff_values.append(pressure_diff)
    #print(f"{t:6.2f} {total_energy:12.4f} {total_momentum:12.4f} {2*ke*mass*(1.57e2)**2/ (N * k_B):12.4e} {pressure:12.4e}")
    T_values.append(T*1.65e-21)
    t += dt

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_values, E_values, label='Total Energy (E)', color='blue')
plt.ylim(np.min(E_values)-10, np.max(E_values)+10)
plt.xlabel('Time (t)')
plt.ylabel('Total Energy (E)')
plt.title('Total Energy vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values, momentum_values, label='Total Momentum', color='red')
plt.ylim(0,10*np.max(momentum_values))
plt.xlabel('Time (t)')
plt.ylabel('Total Momentum')
plt.title('Total Momentum vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_values, T_values, label='Temperature', color='blue')
T_mean = np.mean(T_values)
plt.axhline(y=T_mean, color='red', linestyle='--', label=f'Mean Temperature = {T_mean:.2f}')
plt.xlabel('Time (t)')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_values, P_values, label='Pressure', color='red')
plt.xlabel('Time (t)')
plt.ylabel('Pressure')
plt.title('Pressure vs Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, pressure_diff_values, label="Pressure Difference (Simulation - Ideal)")
plt.ylim(np.min(pressure_diff_values)-1,np.max(pressure_diff_values)+1)
plt.xlabel('Time (s)')
plt.ylabel('Pressure Difference (Pa)')
plt.title('Difference Between Simulation Pressure and Ideal Gas Pressure')
plt.legend()
plt.grid(True)
plt.show()
