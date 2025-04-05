#simple 1 Factor CAPM w/ OLS inverse covariance matrix test Comparison 
#               by. K.tomov 

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#data fetch

stock1 = 'ACN'
stockindex = '^GSPC' 

#range
start = pd.Timestamp('2020-01-01')
end = pd.Timestamp('2025-01-01')
data_1 = yf.download(stock1, start=start, end=end)['Close']
data_2 = yf.download(stockindex, start=start, end=end)['Close']

if data_1.empty or data_2.empty:
    raise ValueError("returns DATA Cannot be retrieved")

print("First 5 rows of downloaded data for {}:".format(stock1))
print(data_1.tail())

return_frequency = 'monthly' # choose either 'daily' or 'monthly' returns (result usually pretty similar)

#conditional statement for resampling
if return_frequency.lower() == 'monthly':
    price_a = data_1.resample('ME').last()
    price_m = data_2.resample('ME').last()
else:  # daily returns
    price_a = data_1
    price_m = data_2

data = pd.concat([price_a, price_m], axis=1)
data.columns = ['Inverse Close', 'Market Close']
if data.empty:
    raise ValueError("data frame is empty")

print("\nFirst 5 rows of {} data:".format(return_frequency))
print(data.head())

#Log returns computation
data[['Inverse Ret', 'Market Ret']] = np.log(data[['Inverse Close', 'Market Close']] / data[['Inverse Close', 'Market Close']].shift(1))
data.dropna(inplace=True)
print("\nFirst 5 rows of returns data:")
print(data.head())

# Relationship between market and investment returns
plt.figure(figsize=(13, 9))
plt.axvline(0, color='grey', alpha=0.5)
plt.axhline(0, color='grey', alpha=0.5)
sns.scatterplot(x='Market Ret', y='Inverse Ret', data=data)
plt.xlabel(f'Market {return_frequency.capitalize()} Return: {stockindex}')
plt.ylabel(f'Investment {return_frequency.capitalize()} Return: {stock1}')
plt.title(f'{return_frequency.capitalize()} Returns plot')
plt.show()

# BETA from the covariance and variance matrix (v1)
cov_matrix = data[['Inverse Ret', 'Market Ret']].cov()
beta_capm = cov_matrix.loc['Inverse Ret', 'Market Ret'] / data['Market Ret'].var()
print('Computed Beta from CAPM:', round(beta_capm, 4))

# BETA & Alpha from OLS regression (v2)
beta_linreg, alpha = np.polyfit(data['Market Ret'], data['Inverse Ret'], deg=1)
print('BETA from Linear Regression:', round(beta_linreg, 4))
print('ALPHA >>> (y) intercept:', round(alpha, 3))

plt.figure(figsize=(13, 9))
plt.axvline(0, color='grey', alpha=0.5)
plt.axhline(0, color='grey', alpha=0.5)
sns.scatterplot(x='Market Ret', y='Inverse Ret', data=data, label='Returns')
sns.lineplot(x=data['Market Ret'], y=alpha + data['Market Ret'] * beta_linreg, color='blue', label='CAPM Line')
plt.xlabel(f'Market {return_frequency.capitalize()} Return: {stockindex}')
plt.ylabel(f'Investment {return_frequency.capitalize()} Return: {stock1}')
plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.)
plt.title(f'{return_frequency.capitalize()} Returns and CAPM Regression')
plt.show()

