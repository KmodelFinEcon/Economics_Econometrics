import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

tickers = ["GE", "ETN"]
data = yf.download(tickers, start="2015-01-01", end="2025-04-12")["Close"]
log_returns = np.log(data / data.shift(1)).dropna()

weights = np.array([0.25, 0.75]) #balancing weights
portfolio_returns = log_returns @ weights

kde = gaussian_kde(portfolio_returns)
x_vals = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 1000)
kde_vals = kde(x_vals)

n_samples = 1000000
simulated_returns = np.random.choice(portfolio_returns, size=n_samples, replace=True)

confidence = 0.99 #confidence interval
var_mc = np.quantile(simulated_returns, 1 - confidence)
cvar_mc = simulated_returns[simulated_returns <= var_mc].mean()

plt.figure(figsize=(12, 6))

sns.histplot(portfolio_returns, bins=50, stat="density", color="blue", alpha=0.4, label="Empirical Distribution")
plt.plot(x_vals, kde_vals, color="blue", linewidth=2, label="Kernel Density Formation")

sns.histplot(simulated_returns, bins=100, stat="density", color="green", alpha=0.3, label="Bootstrapped Monte Carlo")

# VaR / CVaR 
plt.axvline(var_mc, color="green", linestyle="--", linewidth=2, label=f"MC 99% VaR: {var_mc:.2%}")
plt.axvline(cvar_mc, color="green", linestyle=":", linewidth=2, label=f"MC 99% CVaR: {cvar_mc:.2%}")

plt.title("Non-Parametric Gaussian KDE Monte Carlo VaR/CVaR (GE-25 + ETN-75 Portfolio): 1 Million return Simulations ")
plt.xlabel("Daily Log-Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Non-Parametric Monte Carlo 99% CI VaR: {var_mc:.4%}")
print(f"Non-Parametric Monte Carlo 99% CI CVaR: {cvar_mc:.4%}")
