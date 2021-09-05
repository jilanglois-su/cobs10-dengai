import pandas as pd
import os

DATA_RAW = "../data/01_raw/"


class DengueDataApi:
    def __init__(self):
        self.__features_train = pd.read_csv(DATA_RAW + "dengue_features_train.csv")

    def get_features_train(self):
        return self.__features_train


if __name__ == "__main__":
    os.chdir('../')
    dda = DengueDataApi()
    df = dda.get_features_train()
    print(df.head())
