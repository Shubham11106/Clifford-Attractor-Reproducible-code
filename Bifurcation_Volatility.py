import numpy as np
import matplotlib.pyplot as plt

def clifford_step(x, y, a, b, c, d):
    """
    Computes a single step of the 2D Clifford chaotic attractor map.
    """
    x_next = np.sin(a * y) + c * np.cos(a * x)
    y_next = np.sin(b * x) + d * np.cos(b * y)
    return x_next, y_next

def run_sensitivity(param_name, param_range, fixed_params, n_steps=2000, last_n=500):
    """
    Sweeps a target parameter over a specified range and computes the 
    resulting steady-state bifurcation points and volatility (std dev).
    """
    vol_results = []
    p_bif_vals = []
    x_bif_vals = []

    for val in param_range:
        current_params = fixed_params.copy()
        current_params[param_name] = val

        x, y = 0.1, 0.1
        # Transient phase to let system settle onto steady state
        for _ in range(n_steps - last_n):
            x, y = clifford_step(x, y, **current_params)

        # Record bifurcation data and calculate steady state values
        temp_x = []
        for _ in range(last_n):
            x, y = clifford_step(x, y, **current_params)
            temp_x.append(x)
            p_bif_vals.append(val)
            x_bif_vals.append(x)
        vol_results.append(np.std(temp_x))

    return param_range, vol_results, p_bif_vals, x_bif_vals

# Simulation settings: 1000 steps for sweep parameter, 500 points for attractor state
res = 1000
last_n = 500

# =====================================================================
# 1. Parameter c Sensitivity (Systemic Resilience)
# =====================================================================
c_range = np.linspace(1.25, 2, res)
fixed_c = {'a': 1.5, 'b': -1.8, 'd': 0.9}
c_vals, c_vol, c_bif, x_bif_c = run_sensitivity('c', c_range, fixed_c, last_n=last_n)

fig_c, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
ax1.scatter(c_bif, x_bif_c, s=0.1, color='green', alpha=0.3)
ax1.set_title("Bifurcation Analysis: Price Resilience (c) vs. Price Deviation (x)", fontsize=16)
ax1.set_ylabel("Price Deviation (x)", fontsize=18)
ax1.grid(True, alpha=0.2)
ax1.tick_params(axis='both', which='major', labelsize=18)

ax2.plot(c_vals, c_vol, color='darkgreen', lw=2)
ax2.set_title("Volatility Sensitivity: Impact of Resilience (c)", fontsize=14)
ax2.set_ylabel("Volatility (Std Dev)", fontsize=18)
ax2.set_xlabel("Systemic Resilience", fontsize=18)
ax2.grid(True, alpha=0.2)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('sensitivity_analysis_c.png', dpi=600)
plt.show()
plt.close()

# =====================================================================
# 2. Parameter b Sensitivity (Sentiment Feedback)
# =====================================================================
b_range = np.linspace(0.0, 2, res)
fixed_b = {'a': 1.5, 'c': 1.6, 'd': 0.9}
b_vals, b_vol, b_bif, x_bif_b = run_sensitivity('b', b_range, fixed_b, last_n=last_n)

fig_b, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
ax1.scatter(b_bif, x_bif_b, s=0.1, color='purple', alpha=0.3)
ax1.set_title("Bifurcation Analysis: Sentiment Feedback (b) vs. Price Deviation (x)", fontsize=16)
ax1.set_ylabel("Price Deviation", fontsize=18)
ax1.grid(True, alpha=0.2)
ax1.tick_params(axis='both', which='major', labelsize=18)

ax2.plot(b_vals, b_vol, color='purple', lw=2)
ax2.set_title("Volatility Sensitivity: Impact of Sentiment Feedback (b)", fontsize=14)
ax2.set_ylabel("Volatility (Std Dev)", fontsize=18)
ax2.set_xlabel("Sentiment Feedback", fontsize=18)
ax2.grid(True, alpha=0.2)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('sensitivity_analysis_b.png', dpi=600)
plt.show()
plt.close()

# =====================================================================
# 3. Parameter d Sensitivity (Sentiment Inertia / Trending Inertia)
# =====================================================================
d_range = np.linspace(0.0, 2, res)
fixed_d = {'a': 1.5, 'c': 1.6, 'b': -1.8}
d_vals, d_vol, d_bif, x_bif_d = run_sensitivity('d', d_range, fixed_d, last_n=last_n)

fig_d, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
ax1.scatter(d_bif, x_bif_d, s=0.1, color='orange', alpha=0.3)
ax1.set_title("Bifurcation Analysis: Sentiment Inertia (d) vs. Price Deviation (x)", fontsize=16)
ax1.set_ylabel("Price Deviation", fontsize=18)
ax1.grid(True, alpha=0.2)
ax1.tick_params(axis='both', which='major', labelsize=18)

ax2.plot(d_vals, d_vol, color='darkorange', lw=2)
ax2.set_title("Volatility Sensitivity: Impact of Sentiment Inertia (d)", fontsize=14)
ax2.set_ylabel("Volatility (Std Dev)", fontsize=18)
ax2.set_xlabel("Trending Inertia", fontsize=18)
ax2.grid(True, alpha=0.2)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('sensitivity_analysis_d.png', dpi=600)
plt.show()
plt.close()

# =====================================================================
# 4. Parameter a Sensitivity (Traditional Bifurcation / Market Sensitivity)
# =====================================================================
a_range = np.linspace(0.0, 2, res)
fixed_a = {'b': -1.8, 'c': 1.6, 'd': 0.9}
a_vals, a_vol, a_bif, x_bif_a = run_sensitivity('a', a_range, fixed_a, last_n=last_n)

fig_a, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
ax1.scatter(a_bif, x_bif_a, s=0.1, color='red', alpha=0.3)
ax1.set_title("Bifurcation Analysis: Market Sensitivity (a) vs. Price Deviation (x)", fontsize=16)
ax1.set_ylabel("Price Deviation", fontsize=18)
ax1.grid(True, alpha=0.2)
ax1.tick_params(axis='both', which='major', labelsize=18)

ax2.plot(a_vals, a_vol, color='darkred', lw=2)
ax2.set_title("Volatility Sensitivity: Impact of Market Sensitivity (a)", fontsize=14)
ax2.set_ylabel("Volatility (Std Dev)", fontsize=18)
ax2.set_xlabel("Market Sensitivity", fontsize=18)
ax2.grid(True, alpha=0.2)
ax2.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('clifford_market_bifurcation.png', dpi=600)
plt.show()
plt.close()

print("High-resolution plots for c, b, d, and a generated successfully.")

