import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.d04_modeling.abstract_model import AbstractModel
import matplotlib.pyplot as plt


class ARX(AbstractModel):
    def __init__(self, x_train, y_train, p=None):
        """
        :param x_train: covariate data
        :param y_train: outcome data
        :param p: AR order
        """
        super(ARX, self).__init__(x_train=x_train, y_train=y_train, bias=True)
        self._num_rows = len(x_train)
        self.__p = p
        self.__res = dict()
        self.__model = dict()
        self.__cities = x_train.index.get_level_values('city').unique()

    def get_model(self, city):
        return self.__model[city]

    def get_model_results(self, city):
        return self.__res[city]

    def fit(self):
        x_train = self.get_x_train(values=False)
        y_train = self.get_y_train(values=False)

        if self.__p is None:
            model = sm.OLS(endog=np.log(y_train+1), exog=x_train)
            for city in self.__cities:
                self.__model[city] = model
                self.__res[city] = model.fit()
        else:
            for city in self.__cities:
                # TODO: Alternative interpolation methods?
                endog, exog = self.format_data_arimax(x_train.loc[city], y_train.loc[city])
                self.__model[city] = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=(self.__p, 0, 0))
                self.__res[city] = self.__model[city].fit(disp=False, maxiter=100)
                # print(res_arx[city].summary())

    def plot_prediction(self, x_data, y_data):
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        c = 0
        for city in self.__cities:
            y_hat = self.predict(city, x_data, y_data)
            plot_data = self.resample(y_data.loc[city]).to_frame()
            plot_data['prediction'] = y_hat
            t = plot_data.index
            ax[c].plot(t, plot_data['total_cases'])
            ax[c].plot(t, plot_data['prediction'])
            ax[c].set_title(city)
            c += 1
        plt.show()

    def get_mae(self, x_data, y_data):
        mae = 0
        n = 0
        for city in self.__cities:
            y_hat = self.predict(city, x_data, y_data)
            y = self.resample(y_data.loc[city])
            mae += (y-y_hat).abs().sum()
            n += len(y)
        return mae / n

    def predict(self, city, x_data, y_data):
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city])
        if self.__p is None:
            y_log_hat = self.__res[city].predict(exog)
        else:
            res = self.__res[city].apply(endog=endog, exog=exog)
            y_log_hat = res.predict()
        return self.inv_transform_endog(y_log_hat)

    def format_data_arimax(self, x_data, y_data):
        endog = self.resample(self.transform_endog(y_data))
        self.add_constant(x_data)
        exog = self.resample(x_data)
        return endog, exog

    @staticmethod
    def transform_endog(y_data):
        return np.log(y_data + 1)

    @staticmethod
    def inv_transform_endog(log_y_data):
        return np.exp(log_y_data) - 1

    @staticmethod
    def resample(df):
        return df.droplevel('year').resample('W-SUN').median().interpolate()

    def get_residuals(self, city, x_data, y_data):
        y_log = self.transform_endog(self.resample(y_data.loc[city]))
        y_log_hat = self.transform_endog(self.predict(city, x_data, y_data))
        residuals = y_log - y_log_hat
        return residuals

    def analyze_residuals(self, x_data, y_data):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
        c = 0
        for city in self.__cities:
            residuals = self.get_residuals(city, x_data, y_data)
            ax[0][c].plot(residuals.index, residuals)
            ax[0][c].set_title(city)
            sm.graphics.tsa.plot_acf(residuals, lags=60, ax=ax[1][c])
            sm.graphics.tsa.plot_pacf(residuals, lags=60, method='ywm', ax=ax[2][c])
            c += 1
        plt.show()

    def insample_model_evaluation(self):
        model_evaluation = pd.DataFrame(columns=['AIC', 'AICc', 'BIC'], index=pd.Index([]))
        for city in self.__cities:
            model_evaluation.at[city, 'AIC'] = self.__res[city].aic
            model_evaluation.at[city, 'BIC'] = self.__res[city].bic
            try:
                model_evaluation.at[city, 'AICc'] = self.__res[city].aicc
            except:
                print('Model results do not support have AICc')

        return model_evaluation


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    os.chdir('../')
    dda = DengueDataApi()
    x1, x2, y1, y2 = dda.split_data(random=False)
    z1, z2, pct_var = dda.get_svd(x1, x2, num_components=5)

    ols_model = ARX(x_train=z1, y_train=y1, p=None)
    ols_model.fit()
    ols_model.plot_prediction(z1, y1)
    ols_model.analyze_residuals(z1, y1)
    print(ols_model.insample_model_evaluation())
    print("MAE OLS: %.4f" % ols_model.get_mae(z2, y2))

    arx_model = ARX(x_train=z1, y_train=y1, p=5)
    arx_model.fit()
    arx_model.plot_prediction(z1, y1)
    arx_model.analyze_residuals(z1, y1)
    print(arx_model.insample_model_evaluation())
    print("MAE ARX: %.4f" % arx_model.get_mae(z2, y2))

