#### GARCH FORCAST OF STOCK RETURNS#####
#       by K.Tomov

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from arch import arch_model
import math

# Set seaborn style
sns.set(style="darkgrid")

# Define the stock symbols and date range
symbols = ['JD', 'WNS', '^GSPC']
start_date = '2021-01-01'
end_date = '2023-10-01'  # Use a date that is not in the future

# Download historical closing prices for the specified symbols and date range
data = yf.download(symbols, start=start_date, end=end_date)['Close']

# Plot the historical adjusted closing prices
data.plot(figsize=(10, 7))
plt.title('Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend(title='Ticker')
plt.show()

# Compute daily percentage returns and drop missing values
returns = data.pct_change().dropna()

# Plot the daily returns
returns.plot(figsize=(10, 7))
plt.title('Daily Returns of Financial Assets')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend(title='Tickers')
plt.show()

# Calculate daily, monthly, and annual volatility for each stock
volatility_results = []
for stock in returns.columns:
    daily_volatility = returns[stock].std()
    monthly_volatility = math.sqrt(21) * daily_volatility
    annual_volatility = math.sqrt(252) * daily_volatility
    volatility_results.append([stock, daily_volatility, monthly_volatility, annual_volatility])

# Fit GARCH models and generate forecasts for each stock
garch_forecasts = {}
for stock in returns.columns:
    # Fit GARCH(1,1) model
    garch_model = arch_model(returns[stock], p=1, q=1, mean='constant', vol='GARCH', dist='normal')
    gm_result = garch_model.fit(disp='off')
    print(f"\nGARCH Model Parameters for {stock}:")
    print(gm_result.params)
    
    # Generate 5-day variance forecast
    gm_forecast = gm_result.forecast(horizon=5)
    garch_forecasts[stock] = gm_forecast.variance.iloc[-1:].values.flatten()

# Plot GARCH forecasts for each stock side by side
plt.figure(figsize=(12, 6))
for i, stock in enumerate(garch_forecasts.keys()):
    plt.subplot(1, 3, i + 1)
    plt.plot(garch_forecasts[stock], marker='o')
    plt.title(f'GARCH Forecast for {stock}')
    plt.xlabel('Horizon (Days)')
    plt.ylabel('Variance')
plt.tight_layout()
plt.show()

# Rolling predictions for each stock
rolling_predictions_all = {}
test_size = 365

for stock in returns.columns:
    rolling_predictions = []
    for i in range(test_size):
        train = returns[stock][:-(test_size - i)]
        model = arch_model(train, p=1, q=1)
        model_fit = model.fit(disp='off')
        pred = model_fit.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))
    rolling_predictions_all[stock] = pd.Series(rolling_predictions, index=returns[stock].index[-test_size:])

# Plot rolling predictions for each stock
plt.figure(figsize=(14, 8))
for i, stock in enumerate(rolling_predictions_all.keys()):
    plt.subplot(3, 1, i + 1)
    plt.plot(returns[stock][-test_size:], label='True Daily Returns')
    plt.plot(rolling_predictions_all[stock], label='Predicted Volatility')
    plt.title(f'{stock} - Rolling Volatility Prediction')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
plt.tight_layout()
plt.show()