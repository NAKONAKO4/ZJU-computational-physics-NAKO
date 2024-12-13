import numpy as np
import matplotlib.pyplot as plt

def lcg(m, a, c, seed, n):
    x = seed
    numbers = []
    for _ in range(n):
        x = (a * x + c) % m
        numbers.append(x / m)
    return numbers

params = [
    (214326, 1807, 45289),
    (244944, 1597, 51749),
    (233280, 1861, 49297),
    (175000, 2661, 26979),
    (121500, 4081, 25673),
    (145800, 3661, 30809),
    (139968, 3877, 29573),
    (214326, 3613, 45289),
    (714025, 1366, 150889),
    (134456, 8121, 28411),
]

seed = 42
num_samples = 2500

random_sequences = []

for m, a, c in params:
    random_sequences.append(lcg(m, a, c, seed, num_samples))

def plot_histograms(sequences, n_cols=2):
    n_rows = len(sequences) // n_cols + (len(sequences) % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axs = axs.flatten()
    for i, seq in enumerate(sequences):
        axs[i].hist(seq, bins=25, alpha=0.7, color='blue', edgecolor='black')
        axs[i].set_title(f"LCG Set {i+1}")
    plt.tight_layout()
    plt.show()

plot_histograms(random_sequences)

def autocorrelation(sequence):
    n = len(sequence)
    mean = np.mean(sequence)
    autocorr = np.correlate(sequence - mean, sequence - mean, mode='full')
    autocorr = autocorr[n - 1:] / autocorr[n - 1]
    return autocorr

plt.figure(figsize=(10, 6))
for i, seq in enumerate(random_sequences):
    acorr = autocorrelation(seq)
    plt.plot(acorr[50:], label=f"Set {i+1}")
plt.ylim([0, 1])
plt.xlim([0, 2500])
plt.title("Autocorrelation of Random Sequences")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()

from scipy.stats import kstest

print("Kolmogorov-Smirnov Test Results:")
for i, seq in enumerate(random_sequences):
    stat, p_value = kstest(seq, 'uniform')
    print(f"Set {i+1}: KS Statistic = {stat:.5f}, p-value = {p_value:.5f}")

def spectrum_analysis_all(sequences):
    n_cols = 2
    n_rows = len(sequences) // n_cols + (len(sequences) % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axs = axs.flatten()

    freq_range = (0, 0.5)
    magnitude_max = 250

    for i, sequence in enumerate(sequences):
        sequence = sequence - np.mean(sequence)
        fft_result = np.fft.fft(sequence)
        magnitude = np.abs(fft_result)
        freq = np.fft.fftfreq(len(sequence))

        axs[i].plot(freq[:len(freq)//2], magnitude[:len(magnitude)//2], color='blue')
        axs[i].set_title(f"Set {i+1}")
        axs[i].set_xlim(freq_range)
        axs[i].set_ylim(0, magnitude_max)
        axs[i].grid()

    for ax in axs[len(sequences):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

spectrum_analysis_all(random_sequences)

def find_period(sequence):
    seen = {}
    for i, value in enumerate(sequence):
        if value in seen:
            return i - seen[value]
        seen[value] = i
    return None

print("Period Lengths:")
for i, seq in enumerate(random_sequences):
    period = find_period(seq)
    print(f"Set {i+1}: Period Length = {period if period else 'No Period Detected'}")