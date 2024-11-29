import numpy as np

def f(x):
    return np.exp(x) / (np.sin(x)**3 + np.cos(x)**3)

def two_point(f, x, h, n):
    if n == 1:
        return (f(x + h) - f(x)) / h
    else:
        return (two_point(f, x + h, h, n - 1) - two_point(f, x, h, n - 1)) / h

def three_point(f, x, h, n):
    if n == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    else:
        return (three_point(f, x + h, h, n - 1) - three_point(f, x - h, h, n - 1)) / (2 * h)

def five_point(f, x, h, n):
    if n == 1:
        return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
    else:
        return (five_point(f, x + h, h, n - 1) - five_point(f, x - h, h, n - 1)) / (2 * h)

x = 0
h = 0.001 # step size. Change here to reduce the error.
derivatives = 5

results_two_point = [two_point(f, x, h, n) for n in range(1, derivatives + 1)]
results_three_point = [three_point(f, x, h, n) for n in range(1, derivatives + 1)]
results_five_point = [five_point(f, x, h, n) for n in range(1, derivatives + 1)]

print("Two-Point Derivatives:", results_two_point)
print("Three-Point Derivatives:", results_three_point)
print("Five-Point Derivatives:", results_five_point)