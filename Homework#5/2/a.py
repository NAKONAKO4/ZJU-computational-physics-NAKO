import numpy as np
import matplotlib.pyplot as plt
from itertools import product

J = 1.0
L_values = [2,3,4,5]
T_range = np.linspace(1.6, 3.3, 10)


def compute_ising_properties(L, T_range):

    N = L * L
    print(1)
    states = np.array(list(product([-1, 1], repeat=N)))
    print(states)
    E_values = []
    M_values = []

    for state in states:
        state = state.reshape(L, L)
        E = 0
        M = np.sum(state)
        for i in range(L):
            for j in range(L):
                E -= J * state[i, j] * (state[(i + 1) % L, j] + state[i, (j + 1) % L])
        E_values.append(E)
        M_values.append(M)
    E_values = np.array(E_values)
    M_values = np.array(M_values)

    results = {"Cv": [], "m_mean": [], "sqrt_m2": [], "m_max": [], "chi": []}
    for T in T_range:
        beta = 1 / T
        Z = np.sum(np.exp(-beta * E_values))
        E_mean = np.sum(E_values * np.exp(-beta * E_values)) / Z
        E2_mean = np.sum((E_values ** 2) * np.exp(-beta * E_values)) / Z
        M_mean = np.sum(np.abs(M_values) * np.exp(-beta * E_values)) / Z
        M2_mean = np.sum((M_values ** 2) * np.exp(-beta * E_values)) / Z
        M_max = np.max(np.abs(M_values))

        Cv = (E2_mean - E_mean ** 2) * beta ** 2 / N
        chi = (M2_mean - M_mean ** 2) * beta / N

        results["Cv"].append(Cv)
        results["m_mean"].append(M_mean / N)
        results["sqrt_m2"].append(np.sqrt(M2_mean) / N)
        results["m_max"].append(M_max / N)
        results["chi"].append(chi)
    return results


for L in [1]:
    results = compute_ising_properties(5, T_range)
    plt.figure(figsize=(10, 6))

    # 绘制 Cv
    plt.subplot(2, 2, 1)
    plt.plot(T_range, results["Cv"], label=f"L={L}")
    plt.xlabel("T")
    plt.ylabel("$C_v$")
    plt.title("Specific Heat")
    plt.legend()

    # 绘制 <|m|>
    plt.subplot(2, 2, 2)
    plt.plot(T_range, results["m_mean"], label=f"L={L}")
    plt.xlabel("T")
    plt.ylabel("$\\langle |m| \\rangle$")
    plt.title("Mean Magnetization")
    plt.legend()

    # 绘制 sqrt(<m^2>)
    plt.subplot(2, 2, 3)
    plt.plot(T_range, results["sqrt_m2"], label=f"L={L}")
    plt.xlabel("T")
    plt.ylabel("$\\sqrt{\\langle m^2 \\rangle}$")
    plt.title("Root Mean Square Magnetization")
    plt.legend()

    # 绘制 chi
    plt.subplot(2, 2, 4)
    plt.plot(T_range, results["chi"], label=f"L={L}")
    plt.xlabel("T")
    plt.ylabel("$\\chi$")
    plt.title("Susceptibility")
    plt.legend()

    plt.tight_layout()
    plt.show()