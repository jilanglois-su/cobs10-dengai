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

    def fit(self):
        x_train = self.get_x_train(values=False)
        y_train = self.get_y_train(values=False)
        for city in self._cities:
            endog, exog = self.format_data_arimax(x_train.loc[city], y_train.loc[city], interpolate=False)
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
