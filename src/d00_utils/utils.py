import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from src.d01_data.dengue_data_api import WEEK_START_DATE_COL
import statsmodels.api as sm


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


def variable_analysis(x_values, col, ylim=None):
    if ylim is None:
        ylim = [1e-3, 1e2]
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