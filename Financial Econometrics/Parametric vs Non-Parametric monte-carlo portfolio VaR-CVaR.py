# Parametric and Non-parametric Monte-Carlo VaR and CVaR portfolio of assets 
# implementation by K.Tomov
# ****Backtested validation WIP****

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, shapiro, normaltest, anderson
import matplotlib.dates as mdates
np.random.seed(200)

def fetch_data(tickers, startdate, enddate):
    try:
        data = yf.download(tickers, start=startdate, end=enddate)["Close"]
    except Exception as e:
        raise Exception(f"Cannot fetch data: {e}")
    return data

def calc_log_returns(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def calc_portfolio_returns(log_returns, weights):
    portfolio_returns = log_returns @ weights
    return portfolio_returns

def var_cvar(returns, confidence=0.99):
    var = np.quantile(returns, 1 - confidence)
    cvar = returns[returns <= var].mean()
    return var, cvar

def perform_normality_tests(returns): #test for data normality
    results = {}
    shapiro_stat, shapiro_p = shapiro(returns)
    results['Shapiro-Wilk'] = {'statistic': shapiro_stat, 'p-value': shapiro_p}
    dag_stat, dag_p = normaltest(returns)
    results['D\'Agostino-Pearson'] = {'statistic': dag_stat, 'p-value': dag_p}
    anderson_result = anderson(returns)
    results['Anderson-Darling'] = {'statistic': anderson_result.statistic, 'critical_values': anderson_result.critical_values, 'significance_levels': anderson_result.significance_level}
    
    return results

def simulate_bootstrap(returns, n_samples=1000000):
    simulated_returns = np.random.choice(returns, size=n_samples, replace=True)
    return simulated_returns

def simulate_t_distribution(log_returns, weights, num_samples=1000000, df=2):
    mean_vector = log_returns.mean().values
    cov_matrix = log_returns.cov().values
    dim = len(mean_vector)
    
    chol = np.linalg.cholesky(cov_matrix)# Cholesky decomposition for covariance matrix
    
    g = np.random.gamma(df / 2., 2. / df, size=num_samples)
    z = np.random.randn(num_samples, dim)
    
    t_samples = mean_vector + (z @ chol.T) / np.sqrt(g)[:, None]
    portfolio_simulated_t = t_samples @ weights
    return portfolio_simulated_t

def plot_empirical_vs_simulated(empirical_returns, simulated_bootstrap, simulated_t, var_boot, cvar_boot, var_empirical, cvar_empirical, var_t, cvar_t, confidence):
    x_vals = np.linspace(empirical_returns.min(), empirical_returns.max(), 1000)
    kde_empirical = gaussian_kde(empirical_returns)(x_vals)

    plt.figure(figsize=(12, 6))
    sns.histplot(empirical_returns, bins=50, stat="density", color="blue", alpha=0.4, label="Empirical Distribution")
    plt.plot(x_vals, kde_empirical, color="blue", linewidth=2, label="Empirical KDE")
    sns.histplot(simulated_bootstrap, bins=100, stat="density", color="green", alpha=0.3, label="Bootstrapped MC Sim")
    
    plt.axvline(var_boot, color="green", linestyle="--", linewidth=2,label=f"MC {confidence*100:.0f}% VaR: {var_boot:.2%}")
    plt.axvline(cvar_boot, color="green", linestyle=":", linewidth=2,label=f"MC {confidence*100:.0f}% CVaR: {cvar_boot:.2%}")
    plt.title(f"Non-Parametric Monte Carlo VaR/CVaR (At the Confidence: {confidence*100:.0f}%)")
    plt.xlabel("Daily Log-Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(empirical_returns, bins=50, kde=True, stat="density", label="Empirical", color="blue", alpha=0.5)
    sns.histplot(simulated_t, bins=50, kde=True, stat="density", label="Simulated t-Dist", color="red", alpha=0.4)
    
    plt.axvline(var_empirical, color="blue", linestyle="--", 
                label=f"Empirical {confidence*100:.0f}% VaR: {var_empirical:.2%}")
    plt.axvline(cvar_empirical, color="blue", linestyle=":", 
                label=f"Empirical {confidence*100:.0f}% CVaR: {cvar_empirical:.2%}")
    plt.axvline(var_t, color="red", linestyle="--", 
                label=f"t-Sim {confidence*100:.0f}% VaR: {var_t:.2%}")
    plt.axvline(cvar_t, color="red", linestyle=":", 
                label=f"t-Sim {confidence*100:.0f}% CVaR: {cvar_t:.2%}")
    
    plt.title("Portfolio VaR & CVaR: Empirical vs t-Distribution Simulation")
    plt.xlabel("Daily Log-Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def rolling_window_backtest(port_returns, window=250, confidence=0.99):
    rolling_VaRs = []
    violations = []

    for t in range(window, len(port_returns) - 1):
        window_returns = port_returns.iloc[t-window:t]
        VaR, _ = var_cvar(window_returns, confidence)
        rolling_VaRs.append(VaR)
        next_return = port_returns.iloc[t+1]
        violations.append(next_return < VaR)  # True if loss exceeds VaR threshold
    rolling_VaRs = pd.Series(rolling_VaRs, index=port_returns.index[window:len(port_returns)-1])
    
    return violations, rolling_VaRs

# Main execution

if __name__ == "__main__":
    
    tickers = ["MSFT", "WNS"]
    startdate = "2018-01-01"
    enddate = "2023-12-31"
    weights = np.array([0.5, 0.5]) #optimized from chosen portfolio optimization model
    confidence = 0.99
    
    price_data = fetch_data(tickers, startdate, enddate)
    log_return_data = calc_log_returns(price_data)
    port_returns = calc_portfolio_returns(log_return_data, weights)
    
    normality_results = perform_normality_tests(port_returns) #portfolio normality test
    for test, result in normality_results.items():
        print(f"{test} Test:", result)
    
    # Non-parametric Monte Carlo using bootstrapping
    simulated_boot = simulate_bootstrap(port_returns)
    var_boot, cvar_boot = var_cvar(simulated_boot, confidence)
    print(f"Non-Parametric Monte Carlo {confidence*100:.0f}% VaR: {var_boot:.4%}")
    print(f"Non-Parametric Monte Carlo {confidence*100:.0f}% CVaR: {cvar_boot:.4%}")
    
    var_empirical, cvar_empirical = var_cvar(port_returns, confidence)
    
    # Parametric Monte Carlo simulation using t-distribution
    simulated_t = simulate_t_distribution(log_return_data, weights, num_samples=100000, df=5)
    var_t, cvar_t = var_cvar(simulated_t, confidence)
    print(f"Portfolio {confidence*100:.0f}% VaR (Empirical): {var_empirical:.4%}")
    print(f"Portfolio {confidence*100:.0f}% CVaR (Empirical): {cvar_empirical:.4%}")
    print(f"Portfolio {confidence*100:.0f}% VaR (t-Distribution Simulation): {var_t:.4%}")
    print(f"Portfolio {confidence*100:.0f}% CVaR (t-Distribution Simulation): {cvar_t:.4%}")
    
    # Plot the results for both simulation methods
    plot_empirical_vs_simulated(port_returns, simulated_boot, simulated_t, var_boot, cvar_boot, var_empirical, cvar_empirical, var_t, cvar_t, confidence)

    violations, rolling_VaRs = rolling_window_backtest(port_returns, window=250, confidence=0.99)
    violation_rate = np.mean(violations)
    print(f"Back-test violation rate: {violation_rate:.2%}")
    
    plt.figure(figsize=(14, 6))
    plt.plot(port_returns.index[250:len(port_returns)-1], rolling_VaRs, label=f"Rolling {confidence*100:.0f}% VaR", color="red")
    plt.plot(port_returns.index, port_returns, label="Portfolio Returns", alpha=0.6)
    plt.scatter(port_returns.index[250:len(port_returns)-1][np.array(violations)], port_returns.iloc[250:len(port_returns)-1][np.array(violations)], color="black", label="Violations", zorder=5)
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.title("Rolling VaR and Back-Test Violations")
    plt.legend()
    plt.show()
    
    #Extra(yearly VaR/CVaR for reference)
    
    annual_returns = port_returns.resample('YE').sum()  # Aggregates log returns yearly
    annual_var, annual_cvar = var_cvar(annual_returns, confidence)
    print(f"Annual {confidence*100:.0f}% VaR: {annual_var:.4%}")
    print(f"Annual {confidence*100:.0f}% CVaR: {annual_cvar:.4%}")
    
