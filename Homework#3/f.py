import math

def relaxed_iteration(a, x0, lambda_, epsilon, max_iter):
    x_prev = x0
    for i in range(max_iter):
        x_next = x_prev + lambda_ * (math.cos(x_prev) - a * x_prev)  # Relaxed formula
        if abs(x_next - x_prev) < epsilon:
            return x_next, i + 1  # Root and iteration count
        x_prev = x_next
    return None, max_iter

# Example parameters
a = 0.21
x0 = -5  # Initial guess
lambda_ = 0.8  # Relaxation factor
epsilon = 1e-8
max_iter = 1000

root, iterations = relaxed_iteration(a, x0, lambda_, epsilon, max_iter)
if root:
    print(f"Root found: {root:.8f} after {iterations} iterations")
else:
    print("Failed to converge")


def bisection_method(a, x_low, x_high, epsilon, max_iter):
    f = lambda x: math.cos(x) - a * x
    if f(x_low) * f(x_high) > 0:
        raise ValueError("Function has the same sign at both ends of the interval.")

    for i in range(max_iter):
        x_mid = (x_low + x_high) / 2
        if abs(f(x_mid)) < epsilon or abs(x_high - x_low) < epsilon:
            return x_mid, i + 1  # Root and iteration count
        if f(x_low) * f(x_mid) < 0:
            x_high = x_mid
        else:
            x_low = x_mid
    return None, max_iter


# Example parameters
a = 0.21
x_low = -3  # Interval lower bound
x_high = -2  # Interval upper bound
epsilon = 1e-8
max_iter = 1000

root, iterations = bisection_method(a, x_low, x_high, epsilon, max_iter)
if root:
    print(f"Root found: {root:.8f} after {iterations} iterations")
else:
    print("Failed to converge")