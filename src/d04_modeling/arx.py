import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.d04_modeling.abstract_sm import AbstractSM


class ARX(AbstractSM):
    def __init__(self, x_train, y_train, p=None, d=1):
        """
        :param x_train: covariate data
        :param y_train: outcome data
        :param p: AR order
        """
        super(ARX, self).__init__(x_train=x_train, y_train=y_train, bias=True)
        self._num_rows = len(x_train)
        self.__p = p
        self.__d = d

    def fit(self):
        x_train = self.get_x_train(values=False)
        y_train = self.get_y_train(values=False).interpolate()

        if self.__p is None:
            model = sm.OLS(endog=np.log(y_train+1), exog=x_train)
            for city in self._cities:
                self._model[city] = model
                self._res[city] = model.fit()
        else:
            for city in self._cities:
                endog, exog = self.format_data_arimax(x_train.loc[city], y_train.loc[city])
                self._last_obs_t[city] = endog.index[-1]
                self._model[city] = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=(self.__p[city], self.__d, 0))
                self._res[city] = self._model[city].fit(disp=False, maxiter=100)
                # print(res_arx[city].summary())

    def predict(self, city, x_data):
        y_data = pd.Series(np.nan, index=x_data.index, name=self._target_name)
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city])
        if self.__p is None:
            y_log_hat = self._res[city].predict(exog)
        else:
            if endog.index[0] > self._last_obs_t[city]:
                res = self._res[city].extend(endog=endog, exog=exog)
                y_log_hat = res.predict()
            else:
                y_log_hat = self._res[city].predict().reindex(endog.index)
        return self.inv_transform_endog(y_log_hat)

    def forecast(self, city, x_data, y_data, m):
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city])
        if self.__p is None:
            raise Exception("Cant forecast with static model")
        else:
            res = self._res[city]
            y_log_hat = []
            i = 0
            for i in range(len(endog.index)-m):
                forecast = res.get_forecast(m, exog=exog.iloc[i:i+8])
                y_log_hat_m = forecast.conf_int(alpha=0.05)
                y_log_hat_m['total_cases'] = forecast.predicted_mean
                # we are just interested in the last value
                y_log_hat += [y_log_hat_m.iloc[-1].T]
                res = res.extend(endog.iloc[[i]], exog.iloc[[i]])
            forecast = res.get_forecast(m, exog=exog.iloc[i+1:])
            y_log_hat_m = forecast.conf_int(alpha=0.05)
            y_log_hat_m['total_cases'] = forecast.predicted_mean
            # we are just interested in the last value
            y_log_hat += [y_log_hat_m.iloc[-1].T]

            y_log_hat = pd.concat(y_log_hat, axis=1).T
        return self.inv_transform_endog(y_log_hat)


if __name__ == "__main__":
    import os
    from src.d04_modeling.dynamic_factor_model import DynamicFactorModel
    from src.d01_data.dengue_data_api import DengueDataApi
    from collections import defaultdict
    os.chdir('../')
    dda = DengueDataApi()
    x1, x2, y1, y2 = dda.split_data(random=False)
    z1, z2, pct_var = dda.get_svd(x1, x2, num_components=5)

    # dfm_model = DynamicFactorModel(x1.copy(), y1.copy(), factors=3, factor_orders=1, idiosyncratic_ar1=True)
    # dfm_model.fit()
    # z1, y1 = dfm_model.get_filtered_factors(x1, y1)
    # z2, y2 = dfm_model.get_filtered_factors(x2, y2)

    ols_model = ARX(x_train=z1, y_train=y1, p=None)
    ols_model.fit()
    ols_model.plot_prediction(z1, y1)
    ols_model.analyze_residuals(z1, y1)
    print(ols_model.insample_model_evaluation())
    print("MAE OLS: %.4f" % ols_model.get_mae(z2, y2))

    arx_model = ARX(x_train=z1, y_train=y1, p=defaultdict(lambda: 5))
    arx_model.fit()
    arx_model.plot_prediction(z1, y1)
    arx_model.analyze_residuals(z1, y1)
    print(arx_model.insample_model_evaluation())
    print("MAE ARX: %.4f" % arx_model.get_mae(z2, y2))

    arx_model.plot_forecast(z2, y2, m=8)

