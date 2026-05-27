import numpy as np
import matplotlib.pyplot as plt

def clifford_map(x, y, a, b, c, d):
    """
    Computes the next point in the 2D Clifford chaotic attractor map.
    """
    x_next = np.sin(a * y) + c * np.cos(a * x)
    y_next = np.sin(b * x) + d * np.cos(b * y)
    return x_next, y_next

def plot_all_cobwebs(a, b, c, d, n_iter, title_prefix, file_prefix, ic=(0.1, 0.1)):
    """
    Simulates the map trajectory and generates the 2D phase space path 
    and the 1D cobweb projection.
    """
    # Track the trajectory points starting from the initial condition
    x_vals = [ic[0]]
    y_vals = [ic[1]]
    x, y = ic

    for _ in range(n_iter):
        x_new, y_new = clifford_map(x, y, a, b, c, d)
        x_vals.append(x_new)
        y_vals.append(y_new)
        x, y = x_new, y_new

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # --- Plot 1: 2D Iterative Path (Phase Space Plane) ---
    plt.figure(figsize=(10, 10))
    plt.plot(x_vals, y_vals, color='blue', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Color points sequentially to show the temporal flow
    plt.scatter(x_vals, y_vals, c=np.arange(len(x_vals)), cmap='viridis', s=10, zorder=2)
    plt.scatter(x_vals[0], y_vals[0], color='red', marker='*', s=150, label='Start')
    plt.scatter(x_vals[-1], y_vals[-1], color='black', marker='X', s=150, label='End')
    
    plt.title(f'{title_prefix}: 2D Phase Plane Iterative Path', fontsize=14)
    plt.xlabel('$x_n$', fontsize=18)
    plt.ylabel('$y_n$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_2d_path.png', dpi=300)
    plt.show()

    # --- Plot 2: 1D Projection Cobweb (x_{n+1} vs x_n) ---
    plt.figure(figsize=(10, 10))
    xn = x_vals[:-1]
    xn1 = x_vals[1:]

    # Draw the diagonal reference line y = x
    line_min, line_max = -3, 3
    plt.plot([line_min, line_max], [line_min, line_max], color='gray', linestyle='--', label='$x_{n+1}=x_n$')

    # Draw the cobweb staircase steps to visualize convergence/chaos
    for i in range(len(xn)):
        # Vertical step
        plt.plot([xn[i], xn[i]], [xn[i], xn1[i]], color='red', alpha=0.2, linewidth=0.5)
        # Horizontal step
        plt.plot([xn[i], xn1[i]], [xn1[i], xn1[i]], color='blue', alpha=0.2, linewidth=0.5)

    plt.scatter(xn, xn1, s=5, color='black', alpha=0.5, label='Iterates $(x_n, x_{n+1})$')
    plt.title(f'{title_prefix}: 1D Projection Cobweb ($x_{{n+1}}$ vs $x_n$)', fontsize=14)
    plt.xlabel('$x_n$', fontsize=18)
    plt.ylabel('$x_{n+1}$', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_1d_cobweb.png', dpi=300)
    plt.show()

# Base parameters
a, b, d = 1.5, -1.8, 0.9

# 1. Stable system configuration (c = 1.3)
plot_all_cobwebs(a, b, 1.3, d, 100, 'Stable Case (c=1.3)', 'clifford_stable')

# 2. Chaotic system configuration (c = 1.5)
plot_all_cobwebs(a, b, 1.5, d, 300, 'Chaotic Case (c=1.5)', 'clifford_chaotic')

