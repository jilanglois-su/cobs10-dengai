import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from src.d00_utils.constants import WEEK_START_DATE_COL
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss, adfuller


def get_design_matrix(stim, L):
    N = len(stim)
    stim = np.append(np.zeros(L), stim)
    columns = [np.ones(N)]
    for l in range(L):
        new_column = stim[L-(l+1):-(l+1)]
        columns += [new_column]
    return np.stack(columns).T


def log_joint(y, X, w, sigma2):
    ljp = -np.dot(w, w)/(2 * sigma2)
    ljp += np.dot(y, np.dot(X, w))
    ljp -= np.sum(np.exp(np.dot(X, w)))

    return ljp


def log_joint_grad(y, X, w, sigma2):
    return -w/sigma2 + np.dot(X.T, y) - np.dot(X.T, np.exp(np.dot(X, w))).T


def log_joint_hess(y, X, w, sigma2):
    return -np.eye(len(w)) / sigma2 - np.dot(X.T, np.multiply(X, np.exp(np.dot(X, w))[:, np.newaxis]))


def variable_analysis(x_values, col, ax=None, ylim=None):
    if ylim is None:
        ylim = [1e-3, 1e2]
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
    f, Pxx_den = signal.welch(x_values)
    ax[0].semilogy(f, Pxx_den)
    ax[0].set_ylim(ylim)
    ax[0].set_ylabel('Spectral Density')
    ax[0].set_xlabel('$\omega$')
    ax[0].set_title(col)

    t = x_values.index.get_level_values(WEEK_START_DATE_COL)

    ax[1].plot(t, x_values)
    x_smoothed = sm.nonparametric.lowess(x_values, t, frac=0.67, return_sorted=False)
    ax[1].plot(t, x_smoothed)

    plt.show()

    spectral_den = pd.DataFrame(Pxx_den, columns=['power'], index=f)
    spectral_den['t'] = 1./spectral_den.index.to_series()
    return spectral_den.sort_values('power', ascending=False)


def resample2weekly(df, interpolate=True):
    df = df.droplevel('year').resample('W-SUN').median()
    if interpolate:
        return df.interpolate()
    else:
        return df


def kpss_test(timeseries):
    # print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    # print(kpss_output)

    if kpsstest[1] > 0.05:
        # print("KPSS -> stationary")
        return True
    else:
        # print("KPSS -> non-stationary")
        return False


def adf_test(timeseries):
    # print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    # print(dfoutput)

    if dftest[1] > 0.05:
        # print("ADF -> non-stationary")
        return False
    else:
        # print("ADF -> stationary")
        return True
