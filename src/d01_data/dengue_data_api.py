import pandas as pd
import numpy as np
from datetime import timedelta
from src.d00_utils.utils import resample2weekly
from src.d00_utils.constants import *
import os


class DengueDataApi:

    def __init__(self, interpolate=True):
        features_train = pd.read_csv(DATA_RAW + "dengue_features_train.csv", index_col=INDEX_COLS).sort_index()
        features_train[WEEK_START_DATE_COL] = pd.to_datetime(features_train[WEEK_START_DATE_COL])
        features_test = pd.read_csv(DATA_RAW + "dengue_features_test.csv", index_col=INDEX_COLS).sort_index()
        features_test[WEEK_START_DATE_COL] = pd.to_datetime(features_test[WEEK_START_DATE_COL])
        labels_train = pd.read_csv(DATA_RAW + "dengue_labels_train.csv", index_col=INDEX_COLS).sort_index()

        for features_data in [features_test, features_train]:
            for city in features_data.index.get_level_values('city').unique():
                for year in features_data.loc[city].index.get_level_values('year').unique():
                    city_year_data = features_data.loc[city].loc[year]
                    second_to_last_date = city_year_data[WEEK_START_DATE_COL].iloc[-2]
                    last_date = city_year_data[WEEK_START_DATE_COL].iloc[-1]
                    if second_to_last_date > last_date:
                        key = (city, year, city_year_data.index[-1])
                        features_data.at[key, WEEK_START_DATE_COL] = second_to_last_date + timedelta(weeks=1)

        labels_train = labels_train.reindex(features_train.index)
        features_train.reset_index(inplace=True)
        features_train.set_index(['city', 'year', WEEK_START_DATE_COL], inplace=True)
        features_test.reset_index(inplace=True)
        features_test.set_index(['city', 'year', WEEK_START_DATE_COL], inplace=True)
        labels_train.index = features_train.index

        self.__features_train = features_train
        self.__features_test = features_test
        self.__labels_train = labels_train
        x_train = self.__features_train[FEATURE_COLS].copy()
        x_test = self.__features_test[FEATURE_COLS].copy()
        # handle missing values
        if interpolate:
            x_train = x_train.interpolate()
            x_test = x_test.interpolate()
        # transform variables
        x_train[LOG_TRANSFORM] = x_train[LOG_TRANSFORM].apply(lambda x: np.log(x+1))
        x_test[LOG_TRANSFORM] = x_test[LOG_TRANSFORM].apply(lambda x: np.log(x+1))

        # normalize covariates
        self.__x_mean = x_train.mean()
        self.__x_std = x_train.std()
        self.__x_data = self.normalize_x_data(x_train)
        self.__x_test = self.normalize_x_data(x_test)
        self.__y_data = self.__labels_train['total_cases'].interpolate()

    def get_features_train(self):
        return self.__features_train.copy()

    def get_labels_train(self):
        return self.__labels_train.copy()

    def get_features_test(self):
        return self.__features_test.copy()

    def get_x_data(self):
        return self.__x_data.copy()

    def get_y_data(self):
        return self.__y_data.copy()

    def normalize_x_data(self, x_data):
        return (x_data - self.__x_mean.values[np.newaxis, :]) / self.__x_std.values[np.newaxis, :]

    @staticmethod
    def interpolate_nan_(x_data):
        # x_data.sort_index(inplace=True)
        for city in x_data.index.get_level_values('city').unique():
            for year in x_data.loc[city].index.get_level_values('year').unique():
                for col in x_data.columns:
                    interpolated_data = x_data[col].loc[city].loc[year].interpolate()
                    x_data[col].loc[city].loc[year].loc[interpolated_data.index] = interpolated_data.values
        return x_data

    def split_data(self, train_ratio=0.7, seed=1992, random=True):
        x_train = []
        y_train = []
        x_validate = []
        y_validate = []

        np.random.seed(seed=seed)
        idx = pd.IndexSlice
        for city in self.__y_data.index.get_level_values('city').unique():
            year_values = self.__y_data.loc[city].index.get_level_values('year').unique()
            n_train = int(train_ratio * len(year_values))
            if random:
                train_years = pd.Index(np.random.choice(year_values, n_train, replace=False), name=year_values.name)
            else:
                train_years = pd.Index(year_values[:n_train], name=year_values.name)
            validate_years = year_values.difference(train_years)
            x_train += [self.__x_data.loc[idx[city, train_years, :]]]
            x_validate += [self.__x_data.loc[idx[city, validate_years, :]]]
            y_train += [self.__y_data.loc[idx[city, train_years, :]]]
            y_validate += [self.__y_data.loc[idx[city, validate_years, :]]]

        x_train = pd.concat(x_train, axis=0).sort_index()
        y_train = pd.concat(y_train, axis=0).sort_index()
        x_validate = pd.concat(x_validate, axis=0).sort_index()
        y_validate = pd.concat(y_validate, axis=0).sort_index()

        return x_train, x_validate, y_train, y_validate

    @staticmethod
    def get_svd(x_train, x_validate, num_components=4):
        u, s, vh = np.linalg.svd(x_train, full_matrices=True)
        new_features = ["pc%i" % i for i in range(num_components)]
        z_train = pd.DataFrame(np.dot(x_train, vh[:num_components, :].T), columns=new_features, index=x_train.index)
        z_validate = pd.DataFrame(np.dot(x_validate, vh[:num_components, :].T), columns=new_features,
                                  index=x_validate.index)
        var = np.power(s, 2)
        pct_var = (var[:num_components] / var.sum()).sum()
        return z_train, z_validate, pct_var

    @staticmethod
    def get_pca(x_train, x_validate, num_components=4):
        x_cov = x_train.cov()
        w, v = np.linalg.eig(x_cov)
        new_features = ["pc%i" % i for i in range(num_components)]
        z_train = pd.DataFrame(np.dot(x_train, v[:, :num_components]), columns=new_features, index=x_train.index)
        z_validate = pd.DataFrame(np.dot(x_validate, v[:, :num_components]), columns=new_features,
                                  index=x_validate.index)
        var = np.power(w, 2)
        pct_var = (var[:num_components] / var.sum()).sum()
        return z_train, z_validate, pct_var

    def format_data(self, x_data, y_data, interpolate=True):
        endog = self.resample(self.transform_endog(y_data), interpolate)
        for col, diff in diff_variables.items():
            if col in x_data.columns and diff:
                x_data[col] = x_data[col].diff()
                if interpolate:
                    x_data[col].fillna(value=0)

        exog = self.resample(x_data, interpolate)

        return endog, exog

    @staticmethod
    def resample(df, interpolate):
        return resample2weekly(df, interpolate)

    @staticmethod
    def transform_endog(y_data):
        return np.log(y_data + 1).diff()

    @staticmethod
    def inv_transform_endog(dly_data):
        if np.isnan(dly_data.iloc[0]):
            dly_data.iat[0] = 0
        return np.exp(dly_data.cumsum()) - 1


if __name__ == "__main__":
    os.chdir('../')
    dda = DengueDataApi()
    df = dda.get_features_train()
    print(df.head())
