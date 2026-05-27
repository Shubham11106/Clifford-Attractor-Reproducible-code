import numpy as np
import matplotlib.pyplot as plt

def clifford_map(x, y, a, b, c, d):
    """
    Computes a single step of the 2D Clifford attractor map.
    """
    x_next = np.sin(a * y) + c * np.cos(a * x)
    y_next = np.sin(b * x) + d * np.cos(b * y)
    return x_next, y_next

def analyze_sensitivity(a, b, c, d, initial_diff=1e-10, n_iter=150):
    """
    Simulates two trajectories starting from infinitesimally close initial conditions
    and returns the Euclidean distance between them over time (iterations).
    """
    x1, y1 = 0.1, 0.1
    x2, y2 = 0.1 + initial_diff, 0.1
    distances = []
    
    for _ in range(n_iter):
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(dist)
        x1, y1 = clifford_map(x1, y1, a, b, c, d)
        x2, y2 = clifford_map(x2, y2, a, b, c, d)
        
    return np.array(distances)

# --- Global Parameter Configuration ---
a, b, d = 1.5, -1.8, 0.9
c_stable = 1.3
c_threshold = 1.6

# Run trajectory sensitivity simulations
dist_stable = analyze_sensitivity(a, b, c_stable, d)
dist_threshold = analyze_sensitivity(a, b, c_threshold, d)

# --- Plotting Trajectory Divergence (Log Scale) ---
plt.figure(figsize=(12, 6))

# Subplot 1: Stable Regime (Trajectories converge or remain bounded)
plt.subplot(1, 2, 1)
plt.plot(dist_stable, color='blue', lw=1.5, label=f'Stable (c={c_stable})')
plt.yscale('log')
plt.title('Trajectory Divergence (Log Scale): c=1.3')
plt.xlabel('n', fontsize=18)
plt.ylabel('Euclidean Distance', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Subplot 2: Chaotic Regime (Trajectories diverge exponentially - SIC)
plt.subplot(1, 2, 2)
plt.plot(dist_threshold, color='orange', lw=1.5, label=f'Chaotic (c={c_threshold})')
plt.yscale('log')
plt.title('Trajectory Divergence (Log Scale): c=1.6')
plt.xlabel('n', fontsize=18)
plt.ylabel('Euclidean Distance', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Optimize layout and save the figure
plt.tight_layout()
plt.savefig('clifford_sensitivity.png', dpi=300)
plt.show()
plt.close()

# --- Numerical Verification of Chaos (Lyapunov Growth Estimation) ---
# Estimate the exponential growth rate (slope on a log scale) for the chaotic threshold (c=1.6)
log_dist_15 = np.log(dist_threshold[:60])
n_vec = np.arange(60)
slope_15, _ = np.polyfit(n_vec, log_dist_15, 1)

print("Numerical Check:")
print(f"c=1.3 Final Dist: {dist_stable[-1]:.2e}")
print(f"c=1.6 Estimated Growth Rate (Slope): {slope_15:.4f}")
print(f"c=1.6 Final Dist: {dist_threshold[-1]:.2e}")