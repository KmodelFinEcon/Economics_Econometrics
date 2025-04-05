### SIMPLE FROM SAMPLE GAUSSIAN COPULA FOR STOCKS RETURNS ###
#by         K.T.

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

# Collecting data:
tickers = ['BIDU', 'JD', 'VIPS']
data = yf.download(tickers, start='2021-01-01', end='2025-03-01')['Close']

# Daily returns:
returns = data.pct_change().dropna()

# Creation of Gaussian copula
def gaussian_copula(corr_matrix, n_samples=2000):
    mean = [0, 0, 0]
    data = multivariate_normal.rvs(mean, corr_matrix, size=n_samples)
    
    uniform_data = norm.cdf(data)
    return uniform_data

# Computing correlation and sample generation
corr_matrix = returns.corr().values

# Gaussian copula sample generation
samples = gaussian_copula(corr_matrix, n_samples=len(returns))

# Transforming samples to original scale using inverse CDF
returns_BIDU = norm.ppf(samples[:, 0], loc=returns['BIDU'].mean(), scale=returns['BIDU'].std())
returns_JD = norm.ppf(samples[:, 1], loc=returns['JD'].mean(), scale=returns['JD'].std())
returns_VIPS = norm.ppf(samples[:, 2], loc=returns['VIPS'].mean(), scale=returns['VIPS'].std())

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(returns_BIDU, returns_JD, returns_VIPS, alpha=0.5)
ax.set_title('Gaussian Copula of Stock Returns')
ax.set_xlabel('BIDU Returns')
ax.set_ylabel('JD Returns')
ax.set_zlabel('VIPS Returns')
plt.show()