import numpy as np

def f1(x):
    return np.log(x)

def f2(x):
    return np.exp(-x**2)

def f3(x):
    return 1 / (1 + x**2)

def trapezoid_n(a, b, error_tolerance, second_derivative_max):
    return int(np.ceil(np.sqrt(((b - a)**3 * second_derivative_max) / (12 * error_tolerance))))

def simpson_n(a, b, error_tolerance, fourth_derivative_max):
    return int(np.ceil((((b - a)**5 * fourth_derivative_max) / (180 * error_tolerance))**(1/4)))

f1_second_derivative_max = 1
f1_fourth_derivative_max = 1

f2_second_derivative_max = 2
f2_fourth_derivative_max = 12

f3_second_derivative_max = 4
f3_fourth_derivative_max = 24

epsilon1 = 1e-8
epsilon2 = 1e-10
epsilon3 = 1e-12

n1_trap = trapezoid_n(1, 3, epsilon1, f1_second_derivative_max)
n1_simp = simpson_n(1, 3, epsilon1, f1_fourth_derivative_max)

n2_trap = trapezoid_n(-1, 1, epsilon2, f2_second_derivative_max)
n2_simp = simpson_n(-1, 1, epsilon2, f2_fourth_derivative_max)

n3_trap = trapezoid_n(1/2, 5/2, epsilon3, f3_second_derivative_max)
n3_simp = simpson_n(1/2, 5/2, epsilon3, f3_fourth_derivative_max)

print("I1 (log(x)): Trapezoid n =", n1_trap, ", Simpson n =", n1_simp)
print("I2 (exp(-x^2)): Trapezoid n =", n2_trap, ", Simpson n =", n2_simp)
print("I3 (1/(1+x^2)): Trapezoid n =", n3_trap, ", Simpson n =", n3_simp)