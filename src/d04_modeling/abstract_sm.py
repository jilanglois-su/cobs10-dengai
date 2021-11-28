from abc import ABC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from src.d04_modeling.abstract_model import AbstractModel
from src.d00_utils.utils import resample2weekly


class AbstractSM(AbstractModel, ABC):
    def __init__(self, x_train, y_train, bias=True):
        """
        :param x_train: covariate data
        :param y_train: outcome data
        :param bias: add bias term to GLM
        """
        super(AbstractSM, self).__init__(x_train=x_train, y_train=y_train, bias=bias)
        self._res = dict()
        self._model = dict()
        self._cities = x_train.index.get_level_values('city').unique()
        self._last_obs_t = dict()
        self._target_name = y_train.name

    def get_cities(self):
        return self._cities

    def get_model(self, city):
        return self._model[city]

    def get_model_results(self, city):
        return self._res[city]

    @staticmethod
    def resample(df, interpolate=True):
        return resample2weekly(df, interpolate)

    def format_data_arimax(self, x_data, y_data, interpolate=True):
        endog = self.resample(self.transform_endog(y_data), interpolate)
        self.add_constant(x_data)
        exog = self.resample(x_data, interpolate)
        return endog, exog

    @staticmethod
    def transform_endog(y_data):
        return np.log(y_data + 1)

    @staticmethod
    def inv_transform_endog(log_y_data):
        return np.exp(log_y_data) - 1

    def plot_prediction(self, x_data, y_data):
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        c = 0
        for city in self._cities:
            y_hat = self.predict(city, x_data)
            plot_data = self.resample(y_data.loc[city]).to_frame()
            plot_data['prediction'] = y_hat
            t = plot_data.index
            ax[c].plot(t, plot_data['total_cases'])
            ax[c].plot(t, plot_data['prediction'])
            ax[c].set_title(city)
            c += 1
        plt.show()

    def plot_forecast(self, x_data, y_data, m=8):
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        c = 0
        for city in self._cities:
            y_hat = self.forecast(city, x_data, y_data, m)
            plot_data = self.resample(y_data.loc[city]).to_frame()
            plot_data['prediction'] = y_hat[self._target_name]
            plot_data['lower_bound'] = y_hat['lower ' + self._target_name]
            plot_data['upper_bound'] = y_hat['upper ' + self._target_name]
            t = plot_data.index
            ax[c].plot(t, plot_data['total_cases'])
            ax[c].plot(t, plot_data['prediction'])
            ax[c].fill_between(t, plot_data['lower_bound'],
                            plot_data['upper_bound'],
                            facecolor='orange', alpha=0.5)
            ax[c].set_title(city)
            c += 1
        plt.show()

    def get_residuals(self, city, x_data, y_data):
        y_log = self.transform_endog(self.resample(y_data.loc[city]))
        y_log_hat = self.transform_endog(self.predict(city, x_data))
        residuals = y_log - y_log_hat
        return residuals

    def analyze_residuals(self, x_data, y_data):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
        c = 0
        for city in self._cities:
            residuals = self.get_residuals(city, x_data, y_data)
            ax[0][c].plot(residuals.index, residuals)
            ax[0][c].set_title(city)
            sm.graphics.tsa.plot_acf(residuals, lags=60, ax=ax[1][c])
            sm.graphics.tsa.plot_pacf(residuals, lags=60, method='ywm', ax=ax[2][c])
            c += 1
        plt.show()

    def insample_model_evaluation(self):
        model_evaluation = pd.DataFrame(columns=['AIC', 'AICc', 'BIC'], index=pd.Index([]))
        for city in self._cities:
            model_evaluation.at[city, 'AIC'] = self._res[city].aic
            model_evaluation.at[city, 'BIC'] = self._res[city].bic
            try:
                model_evaluation.at[city, 'AICc'] = self._res[city].aicc
            except:
                print('Model results do not support have AICc')

        return model_evaluation

    def get_mae(self, x_data, y_data, m=None):
        mae = 0
        n = 0
        for city in self._cities:
            if m is None:
                y_hat = self.predict(city, x_data)
            else:
                y_hat = self.forecast(city, x_data, y_data, m)[self._target_name]
            y = self.resample(y_data.loc[city])
            mae += (y-y_hat).abs().sum()
            n += len(y)
        return mae / n
