import numpy as np
from autograd import grad


def get_design_matrix(stim, L):
    N = len(stim)
    stim = np.append(np.zeros(L), stim)
    columns = [np.ones(N)]
    for l in range(L):
        new_column = stim[L-(l+1):-(l+1)]
        columns += [new_column]
    return np.stack(columns).T


def log_joint(y, X, w, sigma2):
    ljp = -np.dot(w,w)/(2 * sigma2)
    ljp += np.dot(y, np.dot(X,w))
    ljp -= np.sum(np.exp(np.dot(X,w)))

    return ljp


def log_joint_grad(y, X, w, sigma2):
    return -w/sigma2 + np.dot(X.T, y) - np.dot(X.T, np.exp(np.dot(X, w))).T


def log_joint_hess(y, X, w, sigma2):
    return -np.eye(len(w)) / sigma2 - np.dot(X.T, np.multiply(X, np.exp(np.dot(X, w))[:, np.newaxis]))

