import numpy as np

def initialize_grid_with_guess(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right, guess_type, vmax=20):
    grid = np.zeros((ny, nx))
    grid[0, :] = boundary_top
    grid[-1, :] = boundary_bottom
    grid[:, 0] = boundary_left
    grid[:, -1] = boundary_right

    if guess_type == "average":
        avg_value = (boundary_top + boundary_bottom + boundary_left + boundary_right) / 4
        grid[1:-1, 1:-1] = avg_value
    elif guess_type == "random":
        grid[1:-1, 1:-1] = np.random.uniform(-vmax, vmax, size=(ny-2, nx-2))

    return grid

def jacobi_relaxation(grid, tol=1e-2, max_iter=10000):
    ny, nx = grid.shape
    new_grid = grid.copy()
    for iteration in range(max_iter):
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                new_grid[i, j] = 0.25 * (grid[i+1, j] + grid[i-1, j] + grid[i, j+1] + grid[i, j-1])
        diff = np.abs(new_grid - grid).max()
        if diff < tol:
            return new_grid, iteration
        grid[:, :] = new_grid
    return new_grid, max_iter

nx, ny = 9, 9
boundary_top, boundary_bottom = 9, 9
boundary_left, boundary_right = 5, 5
vmax = 20

# Case 1: Average boundary values guess
grid_average = initialize_grid_with_guess(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right, "average")
final_grid_avg, iterations_avg = jacobi_relaxation(grid_average, tol=0.01)

# Case 2: Random guess
grid_random = initialize_grid_with_guess(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right, "random", vmax=vmax)
final_grid_random, iterations_random = jacobi_relaxation(grid_random, tol=0.01)

print(f"Iterations with average guess: {iterations_avg}")
print(f"Iterations with random guess: {iterations_random}")

difference = final_grid_avg - final_grid_random
abs_difference = np.abs(difference)

mean_diff = np.mean(abs_difference)
max_diff = np.max(abs_difference)

print(f"Mean Absolute Difference of two methods: {mean_diff}")
print(f"Max Absolute Difference of two methods: {max_diff}")