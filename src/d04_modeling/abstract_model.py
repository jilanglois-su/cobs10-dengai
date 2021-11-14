import pandas as pd
import numpy as np

BIAS_COL = 'const'


class AbstractModel:

    def __init__(self, x_train, y_train, bias=True):
        self._w_map = None
        self._bias = bias
        self.add_constant(x_train)
        self._x_train = x_train
        self._y_train = y_train

    def add_constant(self, x_data):
        if self._bias:
            if BIAS_COL not in x_data.columns:
                x_data[BIAS_COL] = 1.
        return None

    def get_x_train(self, values=True):
        is_df = isinstance(self._x_train, pd.DataFrame)
        if values:
            if is_df:
                return self._x_train.values
            else:
                return self._x_train
        else:
            if is_df:
                return self._x_train
            else:
                raise Exception("x Data is not available as DataFrane")

    def get_y_train(self, values=True):
        is_sr = isinstance(self._y_train, pd.Series)
        if values:
            if is_sr:
                return self._y_train.values.reshape((-1, 1))
            else:
                return self._y_train
        else:
            if is_sr:
                return self._y_train
            else:
                raise Exception("y Data is not available as Series")

    def fit(self):
        raise NotImplementedError

    def predict(self, city, x_data):
        raise NotImplementedError

