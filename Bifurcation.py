 import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import linregress
from itertools import permutations
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class CliffordAnalyzer:
    """
    Comprehensive analysis toolkit for Clifford attractor
    """

    def __init__(self, a=1.5, b=-1.8, c=1.6, d=0.9):
        """
        Initialize with default chaotic parameters
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x = 0  # initial condition
        self.y = 0

    def reset(self, x0=0, y0=0):
        """Reset initial conditions"""
        self.x = x0
        self.y = y0

    def step(self):
        """Perform one iteration"""
        x_new = np.sin(self.a * self.y) + self.c * np.cos(self.a * self.x)
        y_new = np.sin(self.b * self.x) + self.d * np.cos(self.b * self.y)
        self.x, self.y = x_new, y_new
        return x_new, y_new

     

    def bifurcation_diagram(self, param_name='c', param_range=np.linspace(1.0, 2.0, 1000),
                           n_steps=1000, n_transient=1000, sample_fraction=0.3):
        """
        Generate bifurcation diagram data
        """
        # Store original parameter
        original_val = getattr(self, param_name)
        bifurcation_data = []

        for p in tqdm(param_range, desc="Bifurcation scan"):
            setattr(self, param_name, p)
            self.reset()

            # Transient
            for _ in range(n_transient):
                self.step()

            # Collect data
            samples = []
            for _ in range(n_steps):
                x, y = self.step()
                samples.append(x)

            # Subsample to avoid overcrowding
            step_size = max(1, int(1/sample_fraction))
            for x_val in samples[::step_size]:
                bifurcation_data.append([p, x_val])

        # Restore parameter
        setattr(self, param_name, original_val)

        return np.array(bifurcation_data)

    def plot_bifurcation(self, bifurcation_data, param_name='c'):
        """Plot bifurcation diagram"""
        plt.figure(figsize=(12, 6))
        plt.scatter(bifurcation_data[:, 0], bifurcation_data[:, 1],
                   s=0.1, alpha=0.3, c='black', marker='.')
        plt.xlabel(f'{param_name}', fontsize=18)
        plt.ylabel('x ', fontsize=18)
        #plt.title(f'Bifurcation Diagram - Clifford Attractor ({param_name} variation)')
        plt.xlim(bifurcation_data[:, 0].min(), bifurcation_data[:, 0].max())
        plt.ylim(-2.5, 2.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return plt.gcf()

# ============================================================
# ONLY BIFURCATION DIAGRAM FOR PARAMETER C
# ============================================================

def bifurcation_only_c():
    """
    Run ONLY the bifurcation diagram for parameter c
    Exactly as in the original code, no other analyses
    """

    # Create analyzer with default parameters
    analyzer = CliffordAnalyzer(a=1.5, b=-1.8, c=1.6, d=0.9)

    # Generate bifurcation data for c (exactly as in original)
    bif_data = analyzer.bifurcation_diagram(
        param_name='c',
        param_range=np.linspace(1.0, 2.0, 1000),  # 500 points as in original
        n_steps=1000,      # 1000 steps per parameter value
        n_transient=1000,   # 500 transient steps
        sample_fraction=0.3  # Keep 30% of points
    )

    # Plot exactly as in original
    fig = analyzer.plot_bifurcation(bif_data, param_name='c')

    # Save the figure
    fig.savefig('clifford_bifurcation_c_only.png', dpi=150)

    plt.show()

    return analyzer, bif_data

# Run ONLY the bifurcation diagram for c
if __name__ == "__main__":
    analyzer, bif_data = bifurcation_only_c()