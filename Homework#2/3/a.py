import numpy as np
import matplotlib.pyplot as plt


def hilbert_matrix(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])


def eigenvalue_ratio_log(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    min_eigenvalue = np.min(np.abs(eigenvalues))
    return np.log(max_eigenvalue / min_eigenvalue)


n_values = range(2, 9)

ratios_single = []
ratios_double = []

for n in n_values:
    H = hilbert_matrix(n)

    # Single precision
    H_single = H.astype(np.float32)
    ratios_single.append(eigenvalue_ratio_log(H_single))

    # Double precision
    H_double = H.astype(np.float64)
    ratios_double.append(eigenvalue_ratio_log(H_double))

plt.figure(figsize=(10, 6))
plt.plot(n_values, ratios_single, marker='o', label="Single Precision")
plt.plot(n_values, ratios_double, marker='s', label="Double Precision")
plt.title("Log of Eigenvalue Ratio (Max/Min) vs Matrix Size")
plt.xlabel("Matrix Size (n)")
plt.ylabel("Log(Max Eigenvalue / Min Eigenvalue)")
plt.legend()
plt.grid()
plt.show()