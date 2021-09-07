import pandas as pd
import numpy as np
import os

DATA_RAW = "../data/01_raw/"

FEATURE_COLS = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm',
                'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                'station_min_temp_c', 'station_precip_mm']
WEEK_START_DATE_COL = 'week_start_date'
INDEX_COLS = ['city', 'year', 'weekofyear']


class DengueDataApi:

    def __init__(self):
        self.__features_train = pd.read_csv(DATA_RAW + "dengue_features_train.csv", index_col=INDEX_COLS)
        self.__features_test = pd.read_csv(DATA_RAW + "dengue_features_test.csv", index_col=INDEX_COLS)
        self.__labels_train = pd.read_csv(DATA_RAW + "dengue_labels_train.csv", index_col=INDEX_COLS)
        # handle missing values
        x_train = self.__features_train[FEATURE_COLS].interpolate()
        x_test = self.__features_test[FEATURE_COLS].interpolate()
        # normalize covariates
        self.__x_mean = x_train.mean()
        self.__x_std = x_train.std()
        self.__x_data = self.normalize_x_data(x_train)
        self.__x_test = self.normalize_x_data(x_test)
        self.__y_data = self.__labels_train['total_cases'].interpolate()

    def get_features_train(self):
        return self.__features_train

    def get_x_data(self):
        return self.__x_data

    def get_y_data(self):
        return self.__y_data

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

    def split_data(self, train_ratio=0.7, seed=1992):
        x_train = []
        y_train = []
        x_validate = []
        y_validate = []

        np.random.seed(seed=seed)
        idx = pd.IndexSlice
        for city in self.__y_data.index.get_level_values('city').unique():
            year_values = self.__y_data.loc[city].index.get_level_values('year').unique()
            n_train = int(train_ratio * len(year_values))
            train_years = pd.Index(np.random.choice(year_values, n_train, replace=False), name=year_values.name)
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


if __name__ == "__main__":
    os.chdir('../')
    dda = DengueDataApi()
    df = dda.get_features_train()
    print(df.head())
