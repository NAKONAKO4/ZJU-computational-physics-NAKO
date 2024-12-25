import numpy as np
import matplotlib.pyplot as plt

sigma = 3.4e-10
epsilon = 1.65e-21
mass = 6.69e-26
tau = sigma * np.sqrt(mass / epsilon)
k_B = 1.38e-23

dt = 0.005
dt2 = dt**2
N = 18
Lx, Ly = 12.0, 6.0
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
            virial += dx * fx + dy * fy
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

n_t_values = []
left_half = Lx / 2

t = 0.0
while t < 300.0:
    ke, pe, virial = verlet(x, y, vx, vy, ax, ay)

    n_t = np.sum(x < left_half)
    n_t_values.append(n_t)

    t += dt
n_t_values = n_t_values[0:60000]

t_values = np.arange(0, 300.0, dt)
print(len(t_values))

n_t_mean = []

for i in range(len(t_values)):
    n_t_mean.append(np.mean(n_t_values[0:i]))

plt.figure(figsize=(16, 6))
plt.plot(t_values, n_t_values, label='$n(t)$: Number of particles in left half')
plt.xlabel('Time (t)')
plt.ylabel('$n(t)$')
plt.title('Number of Particles in Left Half vs Time')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(t_values, n_t_mean, label='$n(t)$: Average number of particles in left half')
plt.xlabel('Time (t)')
plt.ylabel('$n(t)$')
plt.title('Average Number of Particles in Left Half vs Time')
plt.legend()
plt.grid(True)
plt.show()

print(f"Time-averaged number of particles in left half: {n_t_mean:.2f}")