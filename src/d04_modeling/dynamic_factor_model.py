import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.d04_modeling.abstract_sm import AbstractSM


class DynamicFactorModel(AbstractSM):
    def __init__(self, x_train, y_train, factors=1, factor_orders=1):
        """
        :param x_train: covariate data
        :param y_train: outcome data
        :param factors: integer giving the number of (global) factors, a list with the names of (global) factors
        :param factor_orders: integer describing the order of the vector autoregression (VAR) governing all factor
        block dynamics
        """
        super(DynamicFactorModel, self).__init__(x_train=x_train, y_train=y_train, bias=False)
        self._num_rows = len(x_train)
        self.__factors = factors
        self.__factor_orders = factor_orders
        self.__target_name = y_train.name
        self.__endog_mean = dict()
        self.__endog_std = dict()

    def fit(self):
        x_train = self.get_x_train(values=False)
        y_train = self.get_y_train(values=False)
        for city in self._cities:
            endog, exog = self.format_data_arimax(x_train.loc[city], y_train.loc[city], interpolate=False)
            self.__endog_mean[city] = endog.mean()
            self.__endog_std[city] = endog.std()
            endog = pd.concat([endog.to_frame(), exog], axis=1)
            self._last_obs_t[city] = endog.index[-1]
            # Create a dynamic factor model
            self._model[city] = sm.tsa.DynamicFactorMQ(endog,
                                                       factors=self.__factors,
                                                       factor_orders=self.__factor_orders)
            # Note that mod_dfm is an instance of the DynamicFactorMQ class

            # Fit the model via maximum likelihood, using the EM algorithm
            self._res[city] = self._model[city].fit()
            # Note that res_dfm is an instance of the DynamicFactorMQResults class

        return None

    def predict(self, city, x_data):
        y_data = pd.Series(np.nan, index=x_data.index, name=self._target_name)
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city])
        endog = pd.concat([endog.to_frame(), exog], axis=1)
        if endog.index[0] > self._last_obs_t[city]:
            res = self._res[city].extend(endog=endog)
            y_log_hat = res.predict()[self._target_name]
        else:
            y_log_hat = self._res[city].predict().reindex(endog.index)[self._target_name]
        return self.inv_transform_endog(y_log_hat)

    def forecast(self, city, x_data, y_data, m):
        endog, exog = dfm_model.format_data_arimax(x_data.loc[city], y_data.loc[city], interpolate=False)
        endog_extended = pd.concat([endog.iloc[:m].to_frame(), exog.iloc[:m]], axis=1)
        endog_extended[self.__target_name] = np.nan
        y_log_hat = pd.DataFrame(np.nan, index=endog.index, columns=[self.__target_name,
                                                                     'upper ' + self.__target_name,
                                                                     'lower ' + self.__target_name])
        t = endog_extended.index[-8]
        for i in range(len(endog)-m+1):
            if i > 0:
                endog_extended.at[t, 'total_cases'] = endog.loc[t]
                endog_extended = endog_extended.append(exog.iloc[i+m-1])
            t = endog_extended.index[-8]
            t_m = endog_extended.index[-1]
            res = self._res[city].extend(endog=endog_extended)
            y_log_hat_mean = res.states.filtered['eps_M.total_cases'] * self.__endog_std[city] + self.__endog_mean[city]
            y_log_hat_cov = res.states.filtered_cov['eps_M.total_cases']['eps_M.total_cases'] * (self.__endog_std[city] ** 2)
            y_log_hat.at[t_m, self.__target_name] = y_log_hat_mean.loc[t_m]
            y_log_hat.at[t_m, 'upper ' + self.__target_name] = y_log_hat_mean.loc[t_m] + 1.96 * np.sqrt(y_log_hat_cov.loc[t_m])
            y_log_hat.at[t_m, 'lower ' + self.__target_name] = y_log_hat_cov.loc[t_m] - 1.96 * np.sqrt(y_log_hat_cov.loc[t_m])

        return self.inv_transform_endog(y_log_hat)


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    os.chdir('../')
    dda = DengueDataApi(interpolate=False)
    x1, x2, y1, y2 = dda.split_data(random=False)

    dfm_model = DynamicFactorModel(x1.copy(), y1.copy(), factors=3, factor_orders=3)
    dfm_model.fit()

    city = 'sj'
    res_dfm = dfm_model.get_model_results(city)
    print(res_dfm.summary())

    dfm_model.plot_forecast(x2, y2, m=8)
