import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, norm, gamma

#get data

ticker = "JD"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values * 100 #log returns


#bayesian model parameters using student-t distribution 
def log_likelihood(params, data):
    df, loc, scale = params
    if df <= 2 or scale <= 0: #degree of freedom set to 2
        return -np.inf
    return np.sum(t.logpdf(data, df=df, loc=loc, scale=scale))

def log_prior(params):
    df, loc, scale = params
    log_prior_df = gamma.logpdf(df, a=3, scale=1/0.1)  
    log_prior_loc = norm.logpdf(loc, 0, 2.5)
    log_prior_scale = norm.logpdf(scale, 0, 1) if scale > 0 else -np.inf # Half-normal for scale
    return log_prior_df + log_prior_loc + log_prior_scale

def log_posterior(params, data):
    return log_likelihood(params, data) + log_prior(params)

#Metropolis-Hastings Algorithm
def metropolis_hastings(data, initial_params, n_iter=15000, proposal_sd=[0.5, 0.1, 0.1]):
    current_params = initial_params
    samples = np.zeros((n_iter, 3))
    accepted = 0

    for i in range(n_iter):#new parameters using a Gaussian random walk
        proposed_params = current_params + np.random.normal(0, proposal_sd, 3)

        log_alpha = log_posterior(proposed_params, data) - log_posterior(current_params, data)
        alpha = np.exp(log_alpha)

        if np.random.rand() < alpha:
            current_params = proposed_params
            accepted += 1

        samples[i] = current_params

    acceptance_rate = accepted / n_iter
    print(f"AR: {acceptance_rate:.2f}")
    return samples

# Running the MCMC Algorithm and Analyze Results

initial_params = [2.0, np.mean(returns), np.std(returns)]
samples = metropolis_hastings(returns, initial_params, n_iter=15000)

burn_in = 10000 #burn in iterations
posterior_samples = samples[burn_in:]

plt.figure(figsize=(12, 8))
titles = ['Degrees of Freedom', 'Loc', 'Scale']
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(posterior_samples[:, i])
    plt.title(titles[i])
plt.tight_layout()
plt.show()

df_mean = np.mean(posterior_samples[:, 0])
loc_mean = np.mean(posterior_samples[:, 1])
scale_mean = np.mean(posterior_samples[:, 2])
fitted_params = [df_mean, loc_mean, scale_mean]
print(f"Fitted parameters: df = {df_mean:.2f}, loc = {loc_mean:.4f}, scale = {scale_mean:.4f}")


# VaR and CVaR Using Fitted Distribution

def calculate_var_cvar(alpha, params, n_sim=100000):
    #VaR
    df, loc, scale = params
    sim_returns = t.rvs(df=df, loc=loc, scale=scale, size=n_sim)
    var = np.percentile(sim_returns, 100 * (1 - alpha))
    # CVaR
    tail_losses = sim_returns[sim_returns <= var]
    cvar = np.mean(tail_losses)
    return var, cvar, sim_returns

#VaR and CVaR at different confidence levels
confidence_levels = [0.90, 0.95, 0.97]
results = {}
for alpha in confidence_levels:
    var, cvar, sim_returns = calculate_var_cvar(alpha, fitted_params)
    results[alpha] = {"VaR": var, "CVaR": cvar}
    print(f"At {int(alpha*100)}% confidence level: VaR = {var:.2f}%, CVaR = {cvar:.2f}%")

plt.figure(figsize=(10, 7))
plt.hist(sim_returns, bins=50, density=True, alpha=0.5, label="Simulated Returns")

# Confidence lines plot
colors = {0.90: 'red', 0.95: 'green', 0.97: 'blue'}
for alpha in confidence_levels:
    var = results[alpha]["VaR"]
    cvar = results[alpha]["CVaR"]
    plt.axvline(x=var, color=colors[alpha], linestyle="--", 
                label=f"VaR {int(alpha*100)}% ({var:.2f}%)")
    plt.axvline(x=cvar, color=colors[alpha], linestyle="-.", 
                label=f"CVaR {int(alpha*100)}% ({cvar:.2f}%)")

plt.xlabel("Daily Return (%)")
plt.ylabel("Density")
plt.title("Simulated Returns Distribution with VaR and CVaR")
plt.legend()
plt.show()