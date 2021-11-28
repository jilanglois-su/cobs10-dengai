import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.d04_modeling.abstract_sm import AbstractSM
from src.d01_data.dengue_data_api import WEEK_START_DATE_COL


class DynamicFactorModel(AbstractSM):
    def __init__(self, x_train, y_train, factors=1, factor_orders=1, idiosyncratic_ar1=True, inlcude_endog=False):
        """
        :param x_train: covariate data
        :param y_train: outcome data
        :param factors: integer giving the number of (global) factors, a list with the names of (global) factors
        :param factor_orders: integer describing the order of the vector autoregression (VAR) governing all factor
        block dynamics
        :param idiosyncratic_ar1: whether or not to model the idiosyncratic component for each series as an AR(1)
        process. If False, the idiosyncratic component is instead modeled as white noise.
        :param inlcude_endog: whether to include the endog variable in the factor model.
        """
        super(DynamicFactorModel, self).__init__(x_train=x_train, y_train=y_train, bias=False)
        self._num_rows = len(x_train)
        self.__factors = factors
        self.__factor_orders = factor_orders
        self.__target_name = y_train.name
        self.__idiosyncratic_ar1 = idiosyncratic_ar1
        self.__include_endog = inlcude_endog

    def fit(self):
        x_train = self.get_x_train(values=False)
        y_train = self.get_y_train(values=False)
        for city in self._cities:
            endog, exog = self.format_data_arimax(x_train.loc[city], y_train.loc[city], interpolate=False)
            if self.__include_endog:
                endog = pd.concat([endog.to_frame(), exog], axis=1)
            else:
                endog = exog
            self._last_obs_t[city] = endog.index[-1]
            # Create a dynamic factor model
            self._model[city] = sm.tsa.DynamicFactorMQ(endog,
                                                       factors=self.__factors,
                                                       factor_orders=self.__factor_orders,
                                                       idiosyncratic_ar1=self.__idiosyncratic_ar1)
            # Note that mod_dfm is an instance of the DynamicFactorMQ class

            # Fit the model via maximum likelihood, using the EM algorithm
            self._res[city] = self._model[city].fit()
            # Note that res_dfm is an instance of the DynamicFactorMQResults class

        return None

    def predict(self, city, x_data):
        y_data = pd.Series(np.nan, index=x_data.index, name=self._target_name)
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city], interpolate=False)
        if self.__include_endog:
            endog = pd.concat([endog.to_frame(), exog], axis=1)
        else:
            endog = exog
        if endog.index[0] > self._last_obs_t[city]:
            res = self._res[city].extend(endog=endog)
            y_log_hat = res.predict()[self._target_name]
        else:
            y_log_hat = self._res[city].predict().reindex(endog.index)[self._target_name]
        return self.inv_transform_endog(y_log_hat)

    def forecast(self, city, x_data, y_data, m):
        if not self.__include_endog:
            raise Exception("endog not included in the model!")
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
            forecast = res.get_prediction()
            y_log_hat_m = forecast.conf_int(alpha=0.05)
            y_log_hat_m['total_cases'] = forecast.predicted_mean['total_cases']
            y_log_hat.at[t_m, self.__target_name] = y_log_hat_m.loc[t_m, self.__target_name]
            y_log_hat.at[t_m, 'upper ' + self.__target_name] = y_log_hat_m.loc[t_m, 'upper ' + self.__target_name]
            y_log_hat.at[t_m, 'lower ' + self.__target_name] = y_log_hat_m.loc[t_m,  'lower ' + self.__target_name]
        return self.inv_transform_endog(y_log_hat)

    def get_filtered_factors(self, x_data, y_data):
        z_data = []
        y_data_resampled = []
        for city in self._cities:
            z_data_city, y_data_city = self.get_filtered_factors_city(city, x_data, y_data)
            z_data += [z_data_city]
            y_data_resampled += [y_data_city]

        return pd.concat(z_data), pd.concat(y_data_resampled)

    def get_filtered_factors_city(self, city, x_data, y_data):
        endog, exog = self.format_data_arimax(x_data.loc[city], y_data.loc[city], interpolate=False)
        y_log_resampled = endog
        if self.__include_endog:
            endog = pd.concat([endog.to_frame(), exog], axis=1)
        else:
            endog = exog
        if endog.index[0] > self._last_obs_t[city]:
            res = self._res[city].extend(endog=endog)
            factors = res.factors['filtered']
        else:
            factors = self._res[city].factors['filtered']
        factors.columns = ['x%i' % i for i in range(self.__factors)]
        factors.index = pd.MultiIndex.from_arrays([[city] * len(factors), factors.index.year, factors.index],
                                                  names=['city', 'year', WEEK_START_DATE_COL])
        y_log_resampled.index = factors.index
        return factors, self.inv_transform_endog(y_log_resampled)


if __name__ == "__main__":
    import os
    from src.d01_data.dengue_data_api import DengueDataApi
    os.chdir('../')
    dda = DengueDataApi(interpolate=False)
    x1, x2, y1, y2 = dda.split_data(random=False)

    dfm_model = DynamicFactorModel(x1.copy(), y1.copy(), factors=3, factor_orders=2, inlcude_endog=False)
    dfm_model.fit()

    city = 'sj'
    res_dfm = dfm_model.get_model_results(city)
    print(res_dfm.summary())
    z1, y1_ = dfm_model.get_filtered_factors(x1, y1)

    print(z1.head())
    print(y1_.head())
    print(y1.head())

    # dfm_model.plot_forecast(x2, y2, m=8)
