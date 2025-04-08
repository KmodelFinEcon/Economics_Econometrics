
#Hidden Markov regime switching GARCH (1,1) model by K.Tomov

import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats
import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

#objective function

def getreturn(ticker: str, period: str, interval: str) -> pd.Series:
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    returns = data.Close.pct_change().dropna()
    return returns


# ---------------------------
# Statistical Analysis
# ---------------------------
def returnstats(asset_return: pd.Series) -> None:
    fig = px.histogram(asset_return, marginal="violin", title="Return Dist")
    print("Stats of the Return Distribution")
    print("-" * 50)
    print(f"Length of the return series: {asset_return.shape}")
    print(f"Mean: {asset_return.mean()*100:.2f}%")
    print(f"Standard Deviation: {asset_return.std()*100:.2f}%")
    print(f"Skew: {asset_return.skew():.4f}")
    print(f"Kurtosis: {asset_return.kurtosis():.4f}")
    fig.show()



#null hypothesis testing

def testdist(asset_return: pd.Series, mode: str, alpha: float = 1e-2) -> None:
    if mode == 'mean':
        t_stat, p = stats.ttest_1samp(asset_return, popmean=0, alternative='two-sided')
    elif mode == 'normal':
        k2, p = stats.normaltest(asset_return)
    else:
        raise ValueError("mean or normal'")
    
    print(f"p = {p:.4g}")
    if p < alpha:
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")

#GARCH (1,1) fitting

def fit_garch(rt: pd.Series):
    am = arch_model(rt, mean="Constant", vol="Garch", p=1, o=0, q=1,
                    dist="skewt", hold_back=None, rescale=True)
    res = am.fit(disp="off")
    return res


def vol_plot(garch_vol: pd.Series, vl: float, sample_vol: float) -> None:
    fig = px.line(garch_vol, title="GARCH(1,1) assumed vol")
    fig.add_hline(y=vl, line_dash="dash", line_color="green", 
                  annotation_text="Long-run vol")
    fig.add_hline(y=sample_vol, line_dash="dash", line_color="red", 
                  annotation_text="Sample vol")
    fig.show()


#Hidden markov state fitting

def fit_HMM_hmmlearn(vol: pd.Series, n_states: int):

    X = np.array(vol).reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=1500, random_state=70)
    model.fit(X)

    hidden_states = model.predict(X)
    post_prob = model.predict_proba(X)
    
    mus = model.means_.ravel()
    sigmas = np.sqrt(model.covars_.ravel())
    transmat = model.transmat_
    
    print("mean estimation:", mus)
    print("STD estimmations", sigmas)
    
    return hidden_states, mus, sigmas, transmat, post_prob, model

#plotting MSAR and vol

def plot_model(dates, vol, post_prob_df, export_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=vol, name="GARCH Volatility", mode='lines', line_shape='hv', yaxis='y1'))
    
    state_colors = {0: "green", 1: "orange", 2: "red"}
    for state in post_prob_df.columns:
        fig.add_trace(go.Scatter(x=dates, y=post_prob_df[state], name=f"Pr(State {state + 1})", mode='lines', line_shape='hv', line=dict(width=0.5, color=state_colors.get(state, "gray")), stackgroup='two', yaxis='y2'))
    
    fig.update_layout(
        title=f"Volatility Regime - {export_label}",
        yaxis=dict(title="Volatility"),
        yaxis2=dict(title="Posterior Probability", overlaying="y1", side="right")
    )
    fig.write_html(f"Volatility_Regime_Classification_{export_label}.html")
    fig.show()

#Execusion 

if __name__ == "__main__":
    rt = getreturn(ticker='MSFT', period='max', interval='1d')
    returnstats(rt)
    
    print("Hypothesis testing on returns (mean test):")
    testdist(rt, mode="mean")
    print("\nHypothesis testing on returns (normality test):")
    testdist(rt, mode="normal")
    
    # Fit the GARCH model
    volatility_model = fit_garch(rt)
    print(volatility_model.summary())
    
    params = volatility_model.params
    const = params["mu"]
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    
    garch_vol = volatility_model.conditional_volatility.round(2) * np.sqrt(252)
    VL = omega / (1 - alpha - beta)
    sigma_L = np.sqrt(VL) * np.sqrt(252)
    sample_sigma = rt.std() * np.sqrt(252) * 100
    vol_plot(garch_vol, sigma_L, sample_sigma)
    
    # Validate the model using VIX
    vix = yf.Ticker("^VIX").history(period='max', interval="1d").Close
    val_data = pd.DataFrame({'VIX': vix, 'GARCH_vol': garch_vol}).dropna()
    fig_val = px.line(val_data, title="VIX vs GARCH Volatility", line_shape='hv')
    fig_val.show()
    
    diff = val_data["VIX"] - val_data["GARCH_vol"]
    fig_hist = px.histogram(diff, marginal="violin", title="Difference between VIX and GARCH Vol")
    fig_hist.show()
    
    print("Mean Absolute Error:", mean_absolute_error(val_data["VIX"], val_data["GARCH_vol"]))
    testdist(diff, mode="mean")
    
    n_states = 4
    hidden_states, mus, sigmas, transmat, post_prob, hmm_model = fit_HMM_hmmlearn(garch_vol, n_states)
    dates = garch_vol.index
    
    hmm_data = pd.DataFrame({
        "date": dates,
        "volatility": garch_vol,
        "hidden_states": hidden_states
    })

    post_prob_df = pd.DataFrame(post_prob, columns=[0, 1, 2])
    hmm_data["date"] = pd.to_datetime(hmm_data["date"])
    hmm_data.sort_values(by="date", inplace=True)
    
    plot_model(hmm_data["date"], hmm_data["volatility"], post_prob_df, "HMM_hmmlearn")
    
    modelfin = sm.tsa.MarkovAutoregression(rt - rt.mean(), k_regimes=3,
                                                order=1, trend="n",
                                                switching_ar=False,
                                                switching_variance=True)
    modelres = modelfin.fit()
    print(modelres.summary())
    
    msar_post_prob = pd.DataFrame(modelres.smoothed_marginal_probabilities)
    plot_model(dates, garch_vol, msar_post_prob, "MSAR")
    
    # Output log likelihoods and transition matrices for comparison
    print("Log-likelihood of HMM:", hmm_model.score(np.array(garch_vol).reshape(-1,1)))
    print("Transition Matrix")
    print(modelfin.regime_transition_matrix(modelres.params))