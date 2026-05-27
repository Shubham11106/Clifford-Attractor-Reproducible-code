import numpy as np
import matplotlib.pyplot as plt

def simulate_market_transition(n_steps=2000, x0=0.1, y0=0.1):
    """
    Simulates a dynamic market transition using the Clifford attractor model
    where the sensitivity parameter 'a' is slowly ramped up over time.
    """
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    
    # Gradually increasing market sensitivity 'a' over time
    a_values = np.linspace(-0.1, 2.0, n_steps)  
    b, c, d = -1.8, 1.6, 0.9
    x[0], y[0] = x0, y0
    
    for n in range(1, n_steps):
        a = a_values[n]
        x[n] = np.sin(a * y[n-1]) + c * np.cos(a * x[n-1])
        y[n] = np.sin(b * x[n-1]) + d * np.cos(b * y[n-1])
        
    return x, y, a_values

# --- Simulation setup ---
n_steps_crash = 3000
x_trans, y_trans, a_vals_trans = simulate_market_transition(n_steps=n_steps_crash)

# --- Plotting the Transition Phase ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# 1. Price Deviation / Market Evolution Plot
ax1.plot(x_trans, color='darkblue', lw=0.7)
ax1.set_title("Market Evolution: Transition from Stability to Chaos/Crash", fontsize=16)
ax1.set_ylabel("Price Deviation", fontsize=18)
ax1.axvline(x=200, color='green', linestyle='--', alpha=0.6)
ax1.text(2, 1.8, "Stable Regime\n(Low Sensitivity)", color='green', fontweight='bold', fontsize=18)
ax1.axvline(x=1830, color='orange', linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.text(1000, -1.8, "Cyclic Volatility\n(Bubbling)", color='orange', fontweight='bold', fontsize=18)
ax1.text(2000, -2.5, "Chaotic Regime\n(Market Crash/Chaos)", color='red', fontweight='bold', fontsize=18)
ax1.grid(True, alpha=0.2)

# 2. Market Sensitivity Parameter ('a') Plot
ax2.plot(a_vals_trans, color='red', lw=1.5)
ax2.set_title("Market Sensitivity Evolution", fontsize=16)
ax2.set_xlabel("$n$", fontsize=18)
ax2.set_ylabel("Market Sensitivity", fontsize=18)
ax2.grid(True, alpha=0.2)
ax2.tick_params(axis='both', which='major', labelsize=18)

# Polish layout and save high-resolution figure
plt.tight_layout()
plt.savefig('clifford_market_crash_transition.png', dpi=600)
plt.show()
plt.close()

print("Market crash transition plot successfully generated at 600 DPI.")

