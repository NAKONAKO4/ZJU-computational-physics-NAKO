import numpy as np
import matplotlib.pyplot as plt

J = 1.0
L_values = [4, 6, 8, 10, 16, 32, 64, 128]
T_range = np.linspace(1.6, 3.3, 20)
MCS = 100
measurements = 50


def metropolis_step(spins, beta, L):

    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        dE = 2 * J * spins[i, j] * (
                spins[(i + 1) % L, j] + spins[i, (j + 1) % L] +
                spins[(i - 1) % L, j] + spins[i, (j - 1) % L]
        )
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1


def monte_carlo_simulation(L, T_range, MCS, measurements):

    N = L * L
    results = {"Cv": [], "m_mean": [], "sqrt_m2": [], "m_max": [], "chi": []}

    for T in T_range:
        beta = 1 / T
        spins = np.random.choice([-1, 1], size=(L, L))
        E_values = []
        M_values = []

        for _ in range(MCS):
            metropolis_step(spins, beta, L)

        for _ in range(measurements):
            metropolis_step(spins, beta, L)
            E = -J * np.sum(spins * (
                    np.roll(spins, 1, axis=0) +
                    np.roll(spins, 1, axis=1)
            )) / 2
            M = np.sum(spins)
            E_values.append(E)
            M_values.append(M)

        E_values = np.array(E_values)
        M_values = np.array(M_values)
        E_mean = np.mean(E_values)
        E2_mean = np.mean(E_values ** 2)
        M_mean = np.mean(np.abs(M_values))
        M2_mean = np.mean(M_values ** 2)

        Cv = beta ** 2 * (E2_mean - E_mean ** 2) / N
        chi = beta * (M2_mean - M_mean ** 2) / N
        m_max = np.max(np.abs(M_values)) / N

        results["Cv"].append(Cv)
        results["m_mean"].append(M_mean / N)
        results["sqrt_m2"].append(np.sqrt(M2_mean) / N)
        results["m_max"].append(m_max)
        results["chi"].append(chi)

    return results


for L in L_values:
    print(f"Running simulation for L = {L}...")
    results = monte_carlo_simulation(L, T_range, MCS, measurements)

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
