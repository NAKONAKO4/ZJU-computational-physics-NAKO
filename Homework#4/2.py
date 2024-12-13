import itertools
import numpy as np

directions = [
    (1, 0),
    (-1, 0),
    (0.5, 1),
    (-0.5, 1),
    (0.5, -1),
    (-0.5, -1)
]


def enumerate_walks(N):
    return list(itertools.product(range(6), repeat=N))

def calculate_trajectory(steps):
    x, y = 0, 0
    trajectory = [(x, y)]
    for step in steps:
        dx, dy = directions[step]
        x += dx
        y += dy
        trajectory.append((x, y))
    return trajectory


def is_non_reversal(steps):
    for i in range(1, len(steps)):
        if steps[i] == (steps[i - 1] + 3) % 6:
            return False
    return True


def is_self_avoiding(trajectory):
    visited = set()
    for point in trajectory:
        if point in visited:
            return False
        visited.add(point)
    return True

def is_self_avoiding_1(trajectory):
    visited = set()
    a=0
    for point in trajectory:
        if point in visited:
            return False
        visited.add(point)
        a += 1
    return True, a



def calculate_statistics(trajectories):
    x_vals = [traj[-1][0] for traj in trajectories]
    y_vals = [traj[-1][1]*np.sqrt(3)/2 for traj in trajectories]
    # times \sqrt{3}/2 to return the direction to real value.

    mean_x = np.mean(x_vals)
    mean_y = np.mean(y_vals)
    mean_x2 = np.mean(np.array(x_vals) ** 2)
    mean_y2 = np.mean(np.array(y_vals) ** 2)
    mean_delta_r2 = mean_x2 + mean_y2 - mean_x ** 2 - mean_y ** 2
    return mean_x, mean_y, mean_delta_r2


def main():
    N = 4
    all_walks = enumerate_walks(N)

    simple_trajectories = [calculate_trajectory(steps) for steps in all_walks]
    #print(simple_trajectories)
    simple_mean_x, simple_mean_y, simple_mean_delta_r2 = calculate_statistics(simple_trajectories)
    print("Simple random walks:")
    print(f"<x(N)> = {simple_mean_x:.3f}, <y(N)> = {simple_mean_y:.3f}, <ΔR^2(N)> = {simple_mean_delta_r2:.3f}")

    non_reversal_walks = [steps for steps in all_walks if is_non_reversal(steps)]
    non_reversal_trajectories = [calculate_trajectory(steps) for steps in non_reversal_walks]
    non_reversal_mean_x, non_reversal_mean_y, non_reversal_mean_delta_r2 = calculate_statistics(
        non_reversal_trajectories)
    print("\nnon-reversal random walks:")
    print(
        f"<x(N)> = {non_reversal_mean_x:.3f}, <y(N)> = {non_reversal_mean_y:.3f}, <ΔR^2(N)> = {non_reversal_mean_delta_r2:.3f}")

    self_avoiding_trajectories = [traj for traj in simple_trajectories if is_self_avoiding(traj)]
    self_avoiding_mean_x, self_avoiding_mean_y, self_avoiding_mean_delta_r2 = calculate_statistics(
        self_avoiding_trajectories)
    print("\nself-avoiding random walks:")
    print(
        f"<x(N)> = {self_avoiding_mean_x:.3f}, <y(N)> = {self_avoiding_mean_y:.3f}, <ΔR^2(N)> = {self_avoiding_mean_delta_r2:.3f}")
    a=0
    for traj in simple_trajectories:
        if is_self_avoiding(traj):
            a+=1
    print(f"Total number of self-avoiding walks: {a}")

if __name__ == "__main__":
    main()
