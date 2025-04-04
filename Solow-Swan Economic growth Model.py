# simple non-calibrated Solow-Swan Model simulation by K.Tomov

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global parameters assumptions

alpha = 0.3      # Capital share in production
A0 = 1          # Initial total factor productivity
s = 0.2         # Savings rate
delta = 0.1     # Depreciation rate
n = 0.02         # Labor growth rate
g = 0.04         # Productivity growth rate
T = 100         # Number of time periods for simulation
K0 = 10          # Initial capital stock
L0 = 100         # Initial labor force

#Cobb Douglas initiation

def cobb_douglas(K, L, A, alpha):
    return A * (K ** alpha) * (L ** (1 - alpha))

def plot_production_surface():
    K_vals = np.linspace(1, 100, 100)
    L_vals = np.linspace(1, 100, 100)
    K_grid, L_grid = np.meshgrid(K_vals, L_vals)
    Y_grid = cobb_douglas(K_grid, L_grid, A0, alpha)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K_grid, L_grid, Y_grid, cmap='magma')
    ax.set_title('Cobb-Douglas Production Function')
    ax.set_xlabel('Capital')
    ax.set_ylabel('Labor')
    ax.set_zlabel('Total Production')
    plt.show()


def simulate_capital_dynamics():
    L = L0
    capital = np.zeros(T)
    production = np.zeros(T)
    capital[0] = K0

    for t in range(1, T): #capital dynamics
        production[t-1] = cobb_douglas(capital[t-1], L, A0, alpha)
        capital[t] = capital[t-1] + s * production[t-1] - delta * capital[t-1]
    production[T-1] = cobb_douglas(capital[T-1], L, A0, alpha)

    plt.figure(figsize=(12, 6))
  
    plt.subplot(1, 2, 1)
    plt.plot(range(T), capital, label='Capital', color='b')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.title('Capital Accumulation in function of time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(T), production, label='Production', color='g')
    plt.xlabel('Time')
    plt.ylabel('Production')
    plt.title('Production in function of time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def simulate_full_model():
    capital = np.zeros(T)
    labor = np.zeros(T)
    productivity = np.zeros(T)
    production = np.zeros(T)
    capital[0] = K0
    labor[0] = L0
    productivity[0] = A0

    # Simulate dynamics over time
    for t in range(1, T):
        production[t-1] = cobb_douglas(capital[t-1], labor[t-1], productivity[t-1], alpha)
        capital[t] = capital[t-1] + s * production[t-1] - (delta + n + g) * capital[t-1]
        labor[t] = labor[t-1] * (1 + n)
        productivity[t] = productivity[t-1] * (1 + g)
    production[T-1] = cobb_douglas(capital[T-1], labor[T-1], productivity[T-1], alpha)
    
    plt.figure(figsize=(13, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(T), capital, label='Capital', color='b')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.title('Capital Accumulation')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(T), production, label='Production', color='g')
    plt.xlabel('Time')
    plt.ylabel('Production')
    plt.title('Total Production')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(range(T), labor, label='Labor', color='r')
    plt.plot(range(T), productivity, label='Productivity', color='purple')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Labor and Productivity in function of time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

#execution

def main():
    print("P1")
    plot_production_surface()
    
    print("P2")
    simulate_capital_dynamics()
    
    print("P3: Full Dynamics")
    simulate_full_model()

if __name__ == "__main__":
    main()
