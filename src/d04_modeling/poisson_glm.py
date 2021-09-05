import numpy as np
from scipy.optimize import minimize


class PoissonGLM:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.sigma2 = None
        self.num_rows = None

    @staticmethod
    def get_design_matrix(stim, L):
        N = len(stim)
        stim = np.append(np.zeros(L), stim)
        columns = [np.ones(N)]
        for l in range(L):
            new_column = stim[L-(l+1):-(l+1)]
        columns += [new_column]
        return np.stack(columns).T

    @staticmethod
    def get_design_matrix(stim, L):
        N = len(stim)
        stim = np.append(np.zeros(L), stim)
        columns = [np.ones(N)]
        for l in range(L):
            new_column = stim[L-(l+1):-(l+1)]
            columns += [new_column]
        return np.stack(columns).T

    @staticmethod
    def log_joint(y, X, w, sigma2):
        ljp = -np.dot(w, w)/(2 * sigma2)
        ljp += np.dot(y, np.dot(X, w))
        ljp -= np.sum(np.exp(np.dot(X, w)))

        return ljp

    @staticmethod
    def log_joint_grad(y, X, w, sigma2):
        return -w/sigma2 + np.dot(X.T, y) - np.dot(X.T, np.exp(np.dot(X, w))).T

    @staticmethod
    def log_joint_hess(y, X, w, sigma2):
        return -np.eye(len(w)) / sigma2 - np.dot(X.T, np.multiply(X, np.exp(np.dot(X, w))[:, np.newaxis]))

    def compute_posterior_mode(self):
        # Minimize the log joint. Normalize by N so it's better scaled.
        obj = lambda w: -self.log_joint(self.y_train, self.x_train, w, self.sigma2) / self.num_rows
        obj_grad = lambda w: -self.log_joint_grad(self.y_train, self.x_train, w, self.sigma2) / self.num_rows

        # Keep track of the weights visited during optimization.
        w_init = np.zeros(self.x_train.shape[1])
        w_hist = [w_init]

        def callback(w):
            w_hist.append(w)

        result = minimize(obj, w_init, jac=obj_grad, callback=callback, method="BFGS")
        w_hist = np.array(w_hist)
        w_map = w_hist[-1]

        return w_map
