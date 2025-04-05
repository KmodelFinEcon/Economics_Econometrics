
##Econometric index bubble indicator by K.tomov##

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Model parameters in lowercase
ticker = '^gspc'
start = '2020-05-01'
end = '2024-03-01'
adf_lags = 3 #Lags for the Augmented Dickey-Fuller Test. 
crit = 1.49 #Critical value of the right-tailed ADF-Test. 

def is_bubble(ticker, start, end, adf_lags=3, crit=1.49):
    prices = yf.download(ticker, start=start, end=end)['Close']
    if prices.empty:
        print("Error in downloading the data")
        return

    r0 = int(len(prices) * 0.1)
    log_prices = np.log(prices.values)
    delta_log_prices = np.diff(log_prices)
    n = len(delta_log_prices)
    GSO = np.array([])

    # Calculate ADF statistics
    for r2 in range(r0, n):
        adfs = np.array([])
        for r1 in range(0, r2 - r0 + 1):
            X_0 = log_prices[r1:r2+1]
            X = pd.DataFrame()
            X[0] = X_0
            for j in range(1, adf_lags + 1): #lagged indicator
                X[j] = np.append(np.zeros(j), delta_log_prices[r1:r2+1 - j])
            X = np.array(X)
            y = delta_log_prices[r1:r2+1]
            reg = sm.OLS(y, sm.add_constant(X)) #constant term for regression
            res = reg.fit()
            adfs = np.append(adfs, res.params[1] / res.bse[1]) # The test statistic is taken as the t-value of the first regressor (lagged level)
        GSO = np.append(GSO, max(adfs))
    index_plot = prices.index[r0+1:]
    
    # Results
    plt.rc('xtick', labelsize=8)
    plt.plot(index_plot, GSO, label='GSO')
    plt.plot(index_plot, np.ones(len(GSO)) * crit, 'r--', label='Critical Value')
    plt.legend()
    plt.title(f"Bubble Detection for {ticker.upper()}")
    plt.xlabel("Date")
    plt.ylabel("BSADF Statistic")
    plt.show()

    # detected dates where bubble is GSO > critical value
    bubble_dates = index_plot[GSO > crit]
    print("Bubble detected on dates:")
    print(bubble_dates)

if __name__ == '__main__':
    is_bubble(ticker, start, end, adf_lags, crit)
    is_bubble('NVDA', start='2022-05-01', end='2025-07-01') #stock bubble indicator test