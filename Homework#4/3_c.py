import numpy as np

def initialize_grid(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right):
    grid = np.zeros((ny, nx))
    grid[0, :] = boundary_top
    grid[-1, :] = boundary_bottom
    grid[:, 0] = boundary_left
    grid[:, -1] = boundary_right
    return grid

def checkerboard_jacobi_relaxation(grid, tol=1e-2, max_iter=10000):
    ny, nx = grid.shape
    new_grid = grid.copy()

    for iteration in range(max_iter):
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if (i + j) % 2 == 0:  # Red point
                    new_grid[i, j] = 0.25 * (grid[i+1, j] + grid[i-1, j] + grid[i, j+1] + grid[i, j-1])

        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if (i + j) % 2 == 1:  # Black point
                    new_grid[i, j] = 0.25 * (new_grid[i+1, j] + new_grid[i-1, j] + new_grid[i, j+1] + new_grid[i, j-1])

        diff = np.abs(new_grid - grid).max()
        if diff < tol:
            return new_grid, iteration

        grid[:, :] = new_grid

    return new_grid, max_iter

nx, ny = 45, 45
boundary_top, boundary_bottom = 9, 9
boundary_left, boundary_right = 5, 5

grid = initialize_grid(nx, ny, boundary_top, boundary_bottom, boundary_left, boundary_right)

final_grid, iterations = checkerboard_jacobi_relaxation(grid, tol=0.01)

print(f"Checkerboard methods' iterations for grid 45x45：{iterations}")