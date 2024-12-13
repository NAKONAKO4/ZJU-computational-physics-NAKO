import numpy as np

def initialize_grid(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right):
    grid = np.zeros((ny, nx))
    grid[0, :] = boundary_top
    grid[-1, :] = boundary_bottom
    grid[:, 0] = boundary_left
    grid[:, -1] = boundary_right
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

# For nx=ny=9
nx_9, ny_9 = 9, 9
boundary_top, boundary_bottom = 9, 9
boundary_left, boundary_right = 5, 5
grid_9x9 = initialize_grid(nx_9, ny_9, boundary_top, boundary_bottom, boundary_left, boundary_right)
final_grid_9x9, iterations_9x9 = jacobi_relaxation(grid_9x9, tol=0.01)

# For nx=ny=45
nx_45, ny_45 = 45, 45
grid_45x45 = initialize_grid(nx_45, ny_45, boundary_top, boundary_bottom, boundary_left, boundary_right)
final_grid_45x45, iterations_45x45 = jacobi_relaxation(grid_45x45, tol=0.01)

# For nx=ny=72
nx_72, ny_72 = 72, 72
grid_72x72 = initialize_grid(nx_72, ny_72, boundary_top, boundary_bottom, boundary_left, boundary_right)
final_grid_72x72, iterations_72x72 = jacobi_relaxation(grid_72x72, tol=0.01)

print(f"Iterations for 9x9 grid: {iterations_9x9}")
print(f"Iterations for 45x45 grid: {iterations_45x45}")
print(f"Iterations for 72x72 grid: {iterations_72x72}")