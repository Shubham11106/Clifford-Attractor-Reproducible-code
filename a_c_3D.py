import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def clifford_step(x, y, a, b, c, d):
    """
    Computes a single step of the 2D Clifford attractor map.
    """
    x_next = np.sin(a * y) + c * np.cos(a * x)
    y_next = np.sin(b * x) + d * np.cos(b * y)
    return x_next, y_next

def calculate_volatility(a, c, b=-1.8, d=0.9, n_steps=1000, last_n=200):
    """
    Calculates the steady-state volatility (standard deviation of the trajectory)
    for a given pair of parameters 'a' and 'c'.
    """
    x, y = 0.1, 0.1
    # Transient phase to let system settle onto the attractor
    for _ in range(n_steps - last_n):
        x, y = clifford_step(x, y, a, b, c, d)

    # Accumulate steady state values
    vals = []
    for _ in range(last_n):
        x, y = clifford_step(x, y, a, b, c, d)
        vals.append(x)
        
    return np.std(vals)

# --- Define the search grid ---
res = 80
a_vals = np.linspace(0.5, 2.0, res)
c_vals = np.linspace(0.0, 2.0, res)
A, C = np.meshgrid(a_vals, c_vals)

# Vectorize the function to quickly compute volatility across the grid
v_calc = np.vectorize(calculate_volatility)
V = v_calc(A, C)

# --- 3D Visualization ---
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Render the 3D surface
surf = ax.plot_surface(A, C, V, cmap='viridis', edgecolor='none', alpha=0.9)

# Labels and titles
ax.set_title("Global Volatility Surface: Sensitivity (a) vs. Resilience (c)", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel("Market Sensitivity", fontsize=18, labelpad=10)
ax.set_ylabel("Price Resilience", fontsize=18, labelpad=10)
ax.set_zlabel("Volatility (Std Dev of x)", fontsize=18, labelpad=10)
ax.tick_params(labelsize=18)

# Colorbar styling
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label("Volatility Level", fontsize=18)
cbar.ax.tick_params(labelsize=18)

# Use plain formatting for colorbar (no scientific offset)
cbar.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False, useMathText=False))

# Optimize the 3D viewing perspective
ax.view_init(elev=30, azim=225)

# Tight layout and save high-resolution figure
plt.tight_layout()
plt.savefig('volatility_surface_ac.png', dpi=600)
plt.show()
plt.close()

print("3D Volatility surface generated successfully.")

