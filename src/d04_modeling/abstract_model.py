import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BIAS_COL = 'bias'


class AbstractModel:

    def __init__(self, x_train, y_train, bias=True):
        self._w_map = None
        self._bias = bias
        if bias:
            x_train[BIAS_COL] = 1.
        self._x_train = x_train
        self._y_train = y_train

    def get_x_train(self):
        if isinstance(self._x_train, pd.DataFrame):
            return self._x_train.values

    def get_y_train(self):
        if isinstance(self._y_train, pd.Series):
            return self._y_train.values.reshape((-1, 1))

    def log_joint(self, y, X, weights):
        raise NotImplementedError

    def obs_map(self, w, X):
        raise NotImplementedError

    @staticmethod
    def log_joint_grad(y, X, weight, w, sigma2):
        raise NotImplementedError

    @staticmethod
    def log_joint_hess(y, X, weigth, w, sigma2):
        raise NotImplementedError

    def get_w_map(self):
        return self._w_map

    def compute_posterior_mode(self):
        raise NotImplementedError

    def sample_posterior_w(self, num_samples):
        raise NotImplementedError

    def get_posterior_predictive_distribution(self, x_validate, y_validate, ncols, num_samples):
        raise NotImplementedError

    def validate_model(self, x_validate, y_validate, weights_validate=None, ncols=100, num_samples=1000, show_posterior_predictive_dist=True):
        if self._bias:
            x_validate[BIAS_COL] = 1.
        if isinstance(x_validate, pd.DataFrame):
            x_validate = x_validate.values
        if isinstance(y_validate, pd.Series):
            y_validate = y_validate.values.reshape((-1, 1))
        if show_posterior_predictive_dist:
            posterior_predictive_distribution = self.get_posterior_predictive_distribution(x_validate, y_validate,
                                                                                           ncols, num_samples)

            fig, axs = plt.subplots()
            cm = axs.imshow(posterior_predictive_distribution[:, :ncols], aspect='auto')
            axs.plot(np.arange(ncols), y_validate[:ncols], 'r+')
            axs.set_xlabel('n')
            axs.set_ylabel('k')
            fig.colorbar(cm)
            plt.show()

        if weights_validate is None:
            weights_validate = np.ones(y_validate.shape).reshape((-1, 1))
        log_joint = self.log_joint(y_validate, x_validate, weights_validate)

        y_hat = self.obs_map(self._w_map, x_validate)
        e = np.abs(y_validate - y_hat)
        mae = e.mean()

        return log_joint, mae
