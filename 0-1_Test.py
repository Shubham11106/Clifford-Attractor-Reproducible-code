import numpy as np
import matplotlib.pyplot as plt

def clifford_map_series(a, b, c, d, n_iter=5000):
    """
    Generates a time series of x-values from the 2D Clifford attractor map.
    """
    x, y = 0.1, 0.1
    series = []
    
    # Transient phase: discard the first 500 steps to let system settle
    for _ in range(500):
        x_next = np.sin(a * y) + c * np.cos(a * x)
        y_next = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_next, y_next
        
    # Data collection phase
    for _ in range(n_iter):
        x_next = np.sin(a * y) + c * np.cos(a * x)
        y_next = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_next, y_next
        series.append(x)
        
    return np.array(series)

def zero_one_test(phi, n_cut=None):
    """
    Implements the 0-1 statistical test for chaos (developed by Gottwald and Melbourne).
    Returns the K-statistic:
      - K close to 0 indicates regular (periodic or quasiperiodic) dynamics.
      - K close to 1 indicates chaotic dynamics.
    """
    N = len(phi)
    if n_cut is None:
        n_cut = N // 10

    # Constant 'c' chosen in (0, pi)
    c = 1.23

    # Compute translation variables p_n and q_n
    p = np.cumsum(phi * np.cos(np.arange(1, N + 1) * c))
    q = np.cumsum(phi * np.sin(np.arange(1, N + 1) * c))

    # Mean Square Displacement (MSD)
    M = np.zeros(n_cut)
    for n in range(1, n_cut + 1):
        diff_p = p[n:] - p[:-n]
        diff_q = q[n:] - q[:-n]
        M[n-1] = np.mean(diff_p**2 + diff_q**2)

    # Compute correlation coefficient of MSD (K-statistic)
    n_vec = np.arange(1, n_cut + 1)
    K = np.corrcoef(n_vec, M)[0, 1]

    return K

# --- Parameter Configuration ---
a, b, d = 1.5, -1.8, 0.9
c_vals = np.linspace(0.5, 2.0, 300)
K_results = []

print("Executing 0-1 Statistical Test for Chaos...")
for c in c_vals:
    series = clifford_map_series(a, b, c, d)
    K = zero_one_test(series)
    K_results.append(K)

# --- Plotting Results ---
plt.figure(figsize=(12, 6))
plt.plot(c_vals, K_results, color='green', lw=1, label='K-statistic')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axhline(1, color='black', lw=0.5, ls='--')
plt.title(f'0-1 Test for Chaos vs Parameter c (a={a}, b={b}, d={d})', fontsize=14)
plt.xlabel('c', fontsize=18)
plt.ylabel('K-statistic', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-0.2, 1.2)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)

# Tight layout and save high-resolution figure
plt.tight_layout()
plt.savefig('clifford_zero_one_test.png', dpi=300)
plt.show()
plt.close()

print("0-1 test plot saved to clifford_zero_one_test.png")