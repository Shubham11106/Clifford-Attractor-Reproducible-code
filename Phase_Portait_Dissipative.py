import numpy as np
import matplotlib.pyplot as plt

def clifford_step(x, y, a, b, c, d):
    """
    Computes a single iteration step for the 2D Clifford attractor.
    """
    x_next = np.sin(a * y) + c * np.cos(a * x)
    y_next = np.sin(b * x) + d * np.cos(b * y)
    return x_next, y_next

def get_trajectory(a, b, c, d, n_steps=200000):
    """
    Generates the map trajectory, discarding the initial transient points.
    """
    xs, ys = np.zeros(n_steps), np.zeros(n_steps)
    xs[0], ys[0] = 0.1, 0.1
    for i in range(1, n_steps):
        xs[i], ys[i] = clifford_step(xs[i-1], ys[i-1], a, b, c, d)
    # Discard the first 5000 transient steps to ensure we plot the attractor
    return xs[5000:], ys[5000:]

# --- Define parameter sets ---
# Set B: Dissipative (Thin) Fractal
params_b = {'a': 1.6, 'b': -0.6, 'c': -1.2, 'd': 1.6}

# Set A: Non-Dissipative (Fat) Attractor
params_a = {'a': 1.5, 'b': -1.8, 'c': 1.6, 'd': 0.9}

# --- Create subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 1. Plot Set B (Dissipative Attractor)
x_b, y_b = get_trajectory(**params_b)
ax1.scatter(x_b, y_b, s=0.01, color='blue', alpha=0.3)
ax1.set_title("Set B: Dissipative (Thin) Fractal\n$D_{KY} \\approx 1.2879$", fontsize=14)
ax1.set_xlabel("x", fontsize=18)
ax1.set_ylabel("y", fontsize=18)
ax1.tick_params(labelsize=18)
ax1.grid(True, alpha=0.1)

# 2. Plot Set A (Non-Dissipative Attractor)
x_a, y_a = get_trajectory(**params_a)
ax2.scatter(x_a, y_a, s=0.01, color='red', alpha=0.3)
ax2.set_title("Set A: Non-Dissipative (Fat) Attractor\n$D_{KY} = 2.0$", fontsize=14)
ax2.set_xlabel("x", fontsize=18)
ax2.set_ylabel("y", fontsize=18)
ax2.tick_params(labelsize=18)
ax2.grid(True, alpha=0.1)

# Fine-tune layout and save the comparison figure
plt.tight_layout()
plt.savefig('fractal_dimension_comparison.png', dpi=600)
plt.show()
plt.close()

print("Fractal dimension comparison figure generated successfully.")

