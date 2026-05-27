import numpy as np
import matplotlib.pyplot as plt

def clifford_lyapunov(a, b, c, d, n_transient=500, n_iter=2000):
    """
    Computes the largest Lyapunov exponent for the Clifford attractor map.
    """
    # Start from a generic initial point
    x, y = 0.1, 0.1

    # Warm-up phase: let the system settle onto the attractor
    for _ in range(n_transient):
        x_next = np.sin(a * y) + c * np.cos(a * x)
        y_next = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_next, y_next

    # Main loop: track the growth of the tangent vector v
    lyapunov = 0.0
    v = np.array([1.0, 0.0])  # initial perturbation vector

    for _ in range(n_iter):
        # Evaluate the Jacobian at the current point (x, y)
        # J = [[dfx/dx, dfx/dy], [dfy/dx, dfy/dy]]
        j11 = -a * c * np.sin(a * x)
        j12 = a * np.cos(a * y)
        j21 = b * np.cos(b * x)
        j22 = -b * d * np.sin(b * y)

        J = np.array([[j11, j12],
                      [j21, j22]])

        # Push the perturbation vector forward
        v_next = J @ v

        # Renormalize to prevent numerical overflow/underflow
        norm = np.linalg.norm(v_next)
        lyapunov += np.log(norm)
        v = v_next / norm

        # Step the system forward
        x_next = np.sin(a * y) + c * np.cos(a * x)
        y_next = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_next, y_next

    # Average the exponent over the iteration trajectory
    return lyapunov / n_iter

# --- Parameter Configuration ---
a, b, d = 1.5, -1.8, 0.9
c_vals = np.linspace(0.5, 2.0, 500)
le_results = []

print("Calculating Lyapunov exponents...")
for c in c_vals:
    le = clifford_lyapunov(a, b, c, d)
    le_results.append(le)

# --- Plotting Results ---
plt.figure(figsize=(12, 6))
plt.plot(c_vals, le_results, color='red', lw=1, label='Largest Lyapunov Exponent')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.title(f'Lyapunov Exponent vs Parameter c (a={a}, b={b}, d={d})')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('c', fontsize=18)
plt.ylabel('Lyapunov Exponent', fontsize=18)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('lyapunov_plot.png', dpi=300)
plt.show()


