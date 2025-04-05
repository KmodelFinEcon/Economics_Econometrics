#Simple VAR model on Fred US GDP and Consumer confidence data implementation by K.Tomov

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import pandas_datareader.data as web

#data fetch timeline

SD = dt.datetime(1980, 1, 1)
ED = dt.datetime(2025, 4, 4)

def fetch_and_preprocess():
    global df
    data = web.DataReader(["GDP", "UMCSENT"], "fred", SD, ED)
    data.columns = ["GDP", "Consumer confidence"]
    data = data.dropna()
    data["GDP"] = data["GDP"].resample("MS").interpolate(method="linear")
    data["GDP growth"] = data["GDP"].pct_change()
    df = data.dropna().drop(["GDP"], axis=1)

#adf test function

def adf_test(series, series_name):
    result = adfuller(series)
    print(f"ADF Stat  for {series_name}: {result[0]:.6f}")
    print(f"p-value for {series_name}: {result[1]:.6f}\n")
    

#VAR model and fitting

def fit_var_model(forecast_steps=36, irf_steps=36):
    var_model = VAR(df)
    var_results = var_model.fit(ic="bic", verbose=True)
    print(var_results.summary())
    var_results.plot_acorr()#Autocorrelation of residuals
    plt.show()
    
    lag_order = var_results.k_ar
    forecast_input = df.values[-lag_order:]
    forecast = var_results.forecast(forecast_input, steps=15) #15 period forcast
    print("Forecast for next 5 periods:\n", forecast)
    
    var_results.plot_forecast(forecast_steps)
    plt.show()
    
    irf = var_results.irf(irf_steps)
    irf.plot(orth=False, figsize=(18, 8))
    plt.show()
    
    fevd = var_results.fevd(5)
    print(fevd.summary())
    
    fevd.plot(figsize=(12, 8))
    plt.show()
    
    return var_results

#execusion 

if __name__ == "__main__":
    
    fetch_and_preprocess()
    print("Data Head:\n", df.head())
    df.info()
    
    # ADF Tests
    adf_test(df["Consumer confidence"], "Consumer Confidence")
    adf_test(df["GDP growth"], "GDP Growth")
    
    ax = df.plot(secondary_y="Consumer confidence", title="GDP Growth & Consumer Confidence")
    ax.set_ylabel("GDP Growth (%)")
    ax.right_ax.set_ylabel("Consumer Confidence Index")
    plt.show()
    
    # Fit the VAR model and generate forecasts and diagnostics
    var_results = fit_var_model()