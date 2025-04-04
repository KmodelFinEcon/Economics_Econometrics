

##geometric brownian motion##

import matplotlib.pyplot as plt
import numpy as np


def simulate_geometric_brownian_motion(S0, T=2, n_iter=1000, mu=0.01, sigma=0.05):
    dt = T / n_iter

    t = np.linspace(0, T, n_iter)

    # Standard normal distribution N(0, 1)
    W = np.random.standard_normal(size=n_iter)

    W = np.cumsum(W) * np.sqrt(dt)

    X = (mu - 0.5 * sigma**2) * t + sigma * W

    S = S0 * np.exp(X)

    return t, S


def plot_simulation(t, S):
    plt.plot(t, S)
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price S(t)")
    plt.title("Geometric Brownian Motion")
    plt.show()


if __name__ == "__main__":
    time, stock = simulate_geometric_brownian_motion(S0=150)
    plot_simulation(time, stock)

#WIENER PROCESS ONLY

def brownian_motion(dt: float = 0.001, x0=0, n_iter=1000):
    """Simulates brownian motion."""

    # For weiner process, W(t) = 0
    W = np.zeros(n_iter + 1)
    t = np.linspace(x0, n_iter, n_iter + 1)

    W[1 : n_iter + 1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n_iter))
    return t, W


def plot_process(t, W):
    """Plots the brownian motion"""

    plt.plot(t, W)
    plt.xlabel("Time(t)")
    plt.ylabel("Weiner-Process W(t)")
    plt.title("Weiner Process")
    plt.show()


if __name__ == "__main__":
    time, data = brownian_motion()
    plot_process(time, data)