#SARIMAX model implementation % Forecast on US industrial production by. K.Tomov

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm 
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas_datareader.data as web

# Define parameter ranges (lag ranges for the optimization algorithm) for SARIMA

p = range(0, 4)
q = range(0, 4)
P = range(0, 4)
Q = range(0, 4)
d = 1
D = 1
s = 4

#data fetch

#timeframe
sd = dt.datetime(1980, 1, 1)
ed = dt.datetime(2025, 4, 4)

# Fetch data from FRED
data = web.DataReader("INDPRO", "fred", sd, ed)
data.columns = ["Industrial Production"]
data.dropna(inplace=True)

# OG time-series
plt.figure(figsize=(13, 8))
plt.plot(data.index, data["Industrial Production"], label="Industrial Production")
plt.title('US Industrial Production Evolution')
plt.ylabel('Index (2017=100)')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.show()

# PACF and ACF for the original series
plt.figure(figsize=(12, 6))
plot_pacf(data['Industrial Production'], lags=24)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

plt.figure(figsize=(12, 6))
plot_acf(data['Industrial Production'], lags=24)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Stationarity Testing and Transformations

# ADF test OG Time-Series
adf_result = adfuller(data['Industrial Production'])
print(f'ADF Statistic (OG): {adf_result[0]}')
print(f'p-value (OG): {adf_result[1]}')

# Log transformation to get stationarity
data['log'] = np.log(data['Industrial Production'])
data['logdiff'] = data['log'].diff()
data.dropna(inplace=True)

#plot of log-diff
plt.figure(figsize=(13, 8))
plt.plot(data['logdiff'], label="Log Difference")
plt.title("Log Difference of Industrial Production")
plt.xlabel("Date")
plt.legend()
plt.show()

data['seasonal_diff'] = data['logdiff'].diff(4)# Seasonal differencing (lag=4)
data.dropna(inplace=True)

plt.figure(figsize=(13, 8))
plt.plot(data['seasonal_diff'], label="Seasonal Diff")
plt.title("Seasonal Differencing of lag 4 of Log Difference")
plt.xlabel("Date")
plt.legend()
plt.show()

# ADF test on seasonally differenced series
adf_result_seasonal = adfuller(data['seasonal_diff'])
print(f'ADF Statistic (Seasonal Diff): {adf_result_seasonal[0]}')
print(f'p-value (Seasonal Diff): {adf_result_seasonal[1]}')

plt.figure(figsize=(12, 6))
plot_pacf(data['seasonal_diff'], lags=24)
plt.title('PACF of Seasonally Differenced Series')
plt.show()

plt.figure(figsize=(12, 6))
plot_acf(data['seasonal_diff'], lags=24)
plt.title('ACF of Seasonally Differenced Series')
plt.show()

#SARIMA optimization function

def optimize_SARIMA(parameters_list, d, D, s, series):
    
    results = []
    for param in tqdm(parameters_list, desc="Optimizing SARIMA"):
        try: 
            model = SARIMAX(series,
                            order=(param[0], d, param[1]),
                            seasonal_order=(param[2], D, param[3], s)
                           ).fit(disp=False)
        except Exception as e:
            continue  # if failed convergence, then skip
        results.append([param, model.aic])
        
    result_df = pd.DataFrame(results, columns=['(p,q)x(P,Q)', 'AIC'])
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df #dataframe with parameters and corresponding AIC


parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(f"combination of parameter is: {len(parameters_list)}")

result_df = optimize_SARIMA(parameters_list, d, D, s, data['seasonal_diff'])
print(result_df)

#fiting SARIMAX for best forcast accuracy (to test if needed)

best_model = SARIMAX(data['seasonal_diff'], order=(0, 1, 2), seasonal_order=(0, 1, 2, 4)).fit(disp=False)
print(best_model.summary())

#summary statistic
best_model.plot_diagnostics(figsize=(15, 12))
plt.show()

data['arima_model'] = best_model.fittedvalues
data['arima_model'].iloc[:5] = np.nan

forecast = best_model.predict(start=len(data), end=len(data) + 8)
forecast_series = pd.concat([data['arima_model'], forecast])

plt.figure(figsize=(15, 7.5))
plt.plot(forecast_series, color='r', label='Forecast')
plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
plt.plot(data['seasonal_diff'], label='Actual (Seasonally Differenced)')
plt.title("SARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.legend()
plt.show()
