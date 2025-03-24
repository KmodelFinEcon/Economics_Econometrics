import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
from arch import arch_model
from ipywidgets import HBox, VBox, Dropdown, Output
from scipy.optimize import minimize
from scipy.stats import t, norm
from math import inf
from IPython.display import display
import bs4 as bs
import requests
import yfinance as yf
import datetime

#Get S&P 500 tickers from Wikipedia using Bs4
resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if cols:
        ticker = cols[0].text.strip().replace('.', '-')  # transform Wiki ticker symbols for yfinance 
        tickers.append(ticker)

#historical close data from yfinance
start = datetime.datetime(2021, 1, 1)
end = datetime.datetime.now()
close_prices = yf.download(tickers, start=start, end=end)['Close']
rets = ((close_prices / close_prices.shift(1)) - 1).dropna(how='all') * 100 #get percentage returns
rets_subset = rets.iloc[:, :5].fillna(method='ffill').dropna()
print("Subset shape:", rets_subset.shape)

#objective function 

def vecl(matrix): #extracts the lower triangle matrix
    lower_matrix = np.tril(matrix, k=-1)
    array_with_zero = np.array(lower_matrix).flatten()
    array_without_zero = array_with_zero[array_with_zero != 0]
    return array_without_zero

def garch_t_to_u(returns, res): #Transform standardized GARCH residuals to uniform variates using t-distribution cdf
    mu = res.params['mu']
    nu = res.params['nu']
    est_r = returns - mu
    h = res.conditional_volatility
    std_res = est_r / h
    udata = t.cdf(std_res, nu)
    return udata

def dcceq(theta, trdata): #DCC MODEL:  dynamic conditional correlation matrices using the DCC model >>>>>> from exitant reference code
    """
    Compute the dynamic conditional correlation matrices using the DCC model.
    trdata: array with shape (T, N) (each row is an observation for N series).
    theta: parameters (a, b)
    Returns:
      - Rt: array of shape (N, N, T) containing correlation matrices,
      - veclRt: array of shape (T, N*(N-1)/2) with lower-triangular vectorized correlations.
    """
    T, N = np.shape(trdata)
    a, b = theta
    if min(a, b) < 0 or max(a, b) > 1 or (a + b) > .999999:
        a = 0.9999 - b

    Qt = np.zeros((N, N, T))
    Qt[:, :, 0] = np.cov(trdata, rowvar=False)
    
    Rt = np.zeros((N, N, T))
    veclRt = np.zeros((T, int(N*(N-1)/2)))
    Rt[:, :, 0] = np.corrcoef(trdata, rowvar=False)
    
    for j in range(1, T):
        Qt[:, :, j] = Qt[:, :, 0] * (1 - a - b)
        Qt[:, :, j] += a * np.matmul(trdata[j-1:j, :].T, trdata[j-1:j, :])
        Qt[:, :, j] += b * Qt[:, :, j-1]
        d = np.sqrt(np.diag(Qt[:, :, j]))
        Rt[:, :, j] = Qt[:, :, j] / np.outer(d, d)
    
    for j in range(T):
        veclRt[j, :] = vecl(Rt[:, :, j].T)
    return Rt, veclRt

def loglike_norm_dcc_copula(theta, udata): #Negative log-likelihood function for the normal DCC copula.
    T, N = udata.shape
    llf = np.zeros(T)
    trdata = norm.ppf(udata)
    Rt, _ = dcceq(theta, trdata)    # compute the dynamic correlation matrices.
    
    for i in range(T):
        det_R = np.linalg.det(Rt[:, :, i])
        inv_R = np.linalg.inv(Rt[:, :, i])
        term1 = -0.5 * np.log(det_R)
        term2 = -0.5 * (trdata[i, :] @ (inv_R - np.eye(N)) @ trdata[i, :].T)
        llf[i] = term1 + term2
        
    return -np.sum(llf) 

def run_garch_on_return(returns, model_parameters): #Fit a GARCH(t) model for each return series.
    udata_list = [] #2D array of uniform variates (T, N) stacked column-wise
    for col in returns.columns:
        am = arch_model(returns[col], dist='t', vol='Garch', p=1, q=1, mean='Constant') # Fit GARCH with t-distributed errors.
        res = am.fit(disp='off')
        short_name = col.split()[0]
        model_parameters[short_name] = res
        udata = garch_t_to_u(returns[col], res)
        udata_list.append(udata)
    return np.column_stack(udata_list), model_parameters

#main computation

#Run GARCH estimation on each series
model_parameters = {}
udata, model_parameters = run_garch_on_return(rets_subset, model_parameters)

#Estimate DCC parameters using optimization. Establish Constraint: a + b <= 1, bounds for a and b.
cons = {'type': 'ineq', 'fun': lambda x: 1 - (x[0] + x[1])}
bnds = ((0, 0.5), (0, 0.9997))
opt_out = minimize(loglike_norm_dcc_copula, [0.01, 0.95], args=(udata,), bounds=bnds, constraints=cons)

print("Optimization Success:", opt_out.success)
print("Optimized Parameters (a, b):", opt_out.x)
llf = loglike_norm_dcc_copula(opt_out.x, udata)
print("Optimized Negative Log-Likelihood:", llf)

#The dynamic conditional correlations. >>>> From reference code
stock_names = [col.split()[0] for col in rets_subset.columns]
corr_name_list = []
for i, name_a in enumerate(stock_names):
    for name_b in stock_names[:i]:
        corr_name_list.append(f"{name_a}-{name_b}")

trdata = norm.ppf(udata)
_, veclRt = dcceq(opt_out.x, trdata)
dcc_corr = pd.DataFrame(veclRt, index=rets_subset.index, columns=corr_name_list)
dcc_plot = px.line(dcc_corr, title='Dynamic Conditional Correlation Plot', width=1000, height=500)
dcc_plot.show()

#GARCH conditional volatilities plot
garch_vol_df = pd.concat(
    [pd.DataFrame(model_parameters[name].conditional_volatility, index=rets_subset.index) for name in stock_names],
    axis=1
)
garch_vol_df.columns = stock_names
garch_vol_plot = px.line(garch_vol_df, title='GARCH Conditional Volatility', width=1000, height=500)
garch_vol_plot.show()

#Cumulative returns plot
cum_returns = np.log((1 + rets_subset/100).cumprod())
cum_plot = px.line(cum_returns, title='Cumulative Returns', width=1000, height=500)
cum_plot.show()

#unconditional correlation
print("Unconditional correlation (MSFT vs AMZN):")
print(rets_subset.loc[:, ['MSFT', 'AMZN']].corr())

#dynamic correlation update
def update_corr_data(change):
    pair = pair_dropdown.value.split('-')
    if len(pair) != 2:
        return
    a1corr = rets_subset.loc[:, pair].corr().iloc[0, 1]
    idx = corr_name_list.index(pair_dropdown.value)
    a1dcc = pd.DataFrame(veclRt[:, idx], index=rets_subset.index, columns=['DCC'])
    a1dcc['Unconditional'] = a1corr
    corr_line_plot = px.line(a1dcc, title='DCC vs Unconditional Correlation for ' + pair_dropdown.value, width=1000, height=500)
    output_graphics.clear_output()
    with output_graphics:
        display(corr_line_plot)

output_graphics = Output()
pair_dropdown = Dropdown(options=[''] + corr_name_list, description="Pair:")
pair_dropdown.observe(update_corr_data, names='value')
VBox([pair_dropdown, output_graphics])