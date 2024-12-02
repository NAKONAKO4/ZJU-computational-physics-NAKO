import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * np.cosh(x)

def df_exact(x):
    return np.cosh(x) + x * np.sinh(x)

def d2f_exact(x):
    return 2 * np.sinh(x) + x

def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def forward_diff_second(f, x, h):
    return (f(x + 2 * h) - 2 * f(x + h) + f(x)) / (h ** 2)

def central_diff_second(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

def richardson_extrapolation(f_diff, f, x, h, order):
    D1 = f_diff(f, x, h)
    D2 = f_diff(f, x, h / 2)
    return (2 ** order * D2 - D1) / (2 ** order - 1)

x = 1
h_values = np.arange(0.5, 0.05, -0.05)
log_h = np.log(h_values)

log_error_fd1 = []
log_error_cd1 = []
log_error_fd2 = []
log_error_cd2 = []
log_error_richardson1 = []
log_error_richardson2 = []

for h in h_values:
    # First derivative
    fd1 = forward_diff(f, x, h)
    cd1 = central_diff(f, x, h)
    richardson1 = richardson_extrapolation(forward_diff, f, x, h, 1)

    log_error_fd1.append(np.log(np.abs(fd1 - df_exact(x))))
    log_error_cd1.append(np.log(np.abs(cd1 - df_exact(x))))
    log_error_richardson1.append(np.log(np.abs(richardson1 - df_exact(x))))

    # Second derivative
    fd2 = forward_diff_second(f, x, h)
    cd2 = central_diff_second(f, x, h)
    richardson2 = richardson_extrapolation(forward_diff_second, f, x, h, 2)

    log_error_fd2.append(np.log(np.abs(fd2 - d2f_exact(x))))
    log_error_cd2.append(np.log(np.abs(cd2 - d2f_exact(x))))
    log_error_richardson2.append(np.log(np.abs(richardson2 - d2f_exact(x))))

# Plot log errors
plt.figure(figsize=(12, 6))

# First derivative
plt.subplot(1, 2, 1)
plt.plot(log_h, log_error_fd1, label="Forward Difference (1st)")
plt.plot(log_h, log_error_cd1, label="Central Difference (1st)")
plt.plot(log_h, log_error_richardson1, label="Richardson Extrapolation (1st)")
plt.xlabel("log(h)")
plt.ylabel("log(Error)")
plt.title("First Derivative Errors")
plt.legend()

# Second derivative
plt.subplot(1, 2, 2)
plt.plot(log_h, log_error_fd2, label="Forward Difference (2nd)")
plt.plot(log_h, log_error_cd2, label="Central Difference (2nd)")
plt.plot(log_h, log_error_richardson2, label="Richardson Extrapolation (2nd)")
plt.xlabel("log(h)")
plt.ylabel("log(Error)")
plt.title("Second Derivative Errors")
plt.legend()

plt.tight_layout()
plt.show()