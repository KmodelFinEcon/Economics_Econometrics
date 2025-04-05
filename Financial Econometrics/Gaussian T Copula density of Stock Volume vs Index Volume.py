
#Gaussian and T-copula correlation plot between stock specific traded volume and index example
#       by K.Tomov


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import yfinance as yf
from scipy.stats import norm, t
from scipy.special import gamma
from mpl_toolkits.mplot3d import Axes3D 

# Global copula parameters
rho = 0.3# correlation parameter for both copulas
nu = 3# degrees of freedom for t copula

# Stock Specific Volume data
ticker = yf.Ticker("MSFT")
stock_hist = ticker.history(period="5y", interval="1d")
stock_volume = stock_hist['Volume']

# S&P 500 Volume data
sp500_ticker = yf.Ticker("^GSPC")
sp500_hist = sp500_ticker.history(period="5y", interval="1d")
sp500_volume = sp500_hist['Volume']

# Combine volumes
combined_volumes = pd.DataFrame({
    'Stock Volume': stock_volume,
    'SP500 Volume': sp500_volume
}).dropna()

print("Summary Data:")
print(combined_volumes.tail())

# CDFs
stock_ecdf = ECDF(combined_volumes['Stock Volume'])
sp500_ecdf = ECDF(combined_volumes['SP500 Volume'])

combined_volumes['Stock CDF'] = combined_volumes['Stock Volume'].apply(stock_ecdf)
combined_volumes['SP500 CDF'] = combined_volumes['SP500 Volume'].apply(sp500_ecdf)

# 2d Copula scatter plot
joint_plot_height = 7
joint_grid = sns.jointplot(x=combined_volumes['Stock CDF'],
                           y=combined_volumes['SP500 CDF'],
                           kind="reg",
                           height=joint_plot_height,
                           xlim=(-0.05, 1.05),
                           ylim=(-0.05, 1.05),
                           joint_kws={"line_kws": {"color": "firebrick"}})
plt.suptitle("individual stock vs S&P 500 Volume Copula (CDF)", fontsize=18)
plt.tight_layout()
plt.show()

#grid in (u,v) space
grid_size = 250
u_vals = np.linspace(0.001, 0.999, grid_size)
v_vals = np.linspace(0.001, 0.999, grid_size)
U, V = np.meshgrid(u_vals, v_vals)

# Inverse CDF transforms for Gaussian copula
Z_u = norm.ppf(U)
Z_v = norm.ppf(V)

# Gaussian Copula
num = np.exp(- (Z_u**2 - 2*rho*Z_u*Z_v + Z_v**2) / (2*(1-rho**2)))# numerator: bivariate normal density with correlation rho
denom = np.exp(- (Z_u**2 + Z_v**2)/2)# denominator: product of standard normal densities (up to constant factors)
gauss_copula_density = (1/np.sqrt(1-rho**2)) * num / denom

# t-Copula Density 
T_u = t.ppf(U, df=nu)# Inverse CDF for t distribution with nu degrees of freedom
T_v = t.ppf(V, df=nu)

#bivariate t-ensity function 
def biv_t_pdf(x, y, nu, rho):
    factor = (gamma((nu+2)/2) /
              (gamma(nu/2) * np.pi * nu * np.sqrt(1-rho**2)))
    quad_form = (x**2 - 2*rho*x*y + y**2) / (nu*(1-rho**2))
    return factor * (1 + quad_form) ** (-(nu+2)/2)

biv_t_density = biv_t_pdf(T_u, T_v, nu, rho)
t_pdf_u = t.pdf(T_u, df=nu)
t_pdf_v = t.pdf(T_v, df=nu)
t_copula_density = biv_t_density / (t_pdf_u * t_pdf_v)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(U, V, gauss_copula_density, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_title("Gaussian Copula")
ax1.set_xlabel("u")
ax1.set_ylabel("v")
ax1.set_zlabel("Density")
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(U, V, t_copula_density, cmap='magma', edgecolor='none', alpha=0.9)
ax2.set_title(f"t Copula (nu={nu})")
ax2.set_xlabel("u")
ax2.set_ylabel("v")
ax2.set_zlabel("Density")
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

plt.suptitle("Copula Density Surfaces", fontsize=15)
plt.tight_layout()
plt.show()