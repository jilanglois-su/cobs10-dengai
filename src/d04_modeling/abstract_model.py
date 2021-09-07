import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BIAS_COL = 'bias'


class AbstractModel:

    def __init__(self, x_train, y_train, bias=True):
        self.__w_map = None
        self.__bias = bias
        if bias:
            x_train[BIAS_COL] = 1.
        self.__x_train = x_train
        self.__y_train = y_train

    def get_x_train(self):
        return self.__x_train

    def get_y_train(self):
        return self.__y_train

    def log_joint(self, y, X):
        raise NotImplementedError

    @staticmethod
    def log_joint_grad(y, X, w, sigma2):
        raise NotImplementedError

    @staticmethod
    def log_joint_hess(y, X, w, sigma2):
        raise NotImplementedError

    def get_w_map(self):
        return self.__w_map

    def compute_posterior_mode(self):
        raise NotImplementedError

    def sample_posterior_w(self, num_samples):
        raise NotImplementedError

    def get_posterior_predictive_distribution(self, x_validate, y_validate, ncols, num_samples):
        raise NotImplementedError

    def validate_model(self, x_validate, y_validate, ncols=100, num_samples=1000, show_posterior_predictive_dist=True):
        if self.__bias:
            x_validate[BIAS_COL] = 1.
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

        log_joint = self.log_joint(y_validate, x_validate)

        return log_joint
