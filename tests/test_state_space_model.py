from unittest import TestCase
import pandas as pd
from src.d01_data.dengue_data_api import DengueDataApi
from src.d04_modeling.abstract_sm import AbstractSM
from src.d04_modeling.state_space_model import StateSpaceModel


class TestStateSpaceModel(TestCase):
    def test_update(self):
        dda = DengueDataApi(interpolate=False)
        x1, x2, y1, y2 = dda.split_data(random=False)

        abstract_model = AbstractSM(x_train=x1, y_train=y1, bias=False)
        for city in abstract_model._cities:
            endog, exog = abstract_model.format_data_arimax(x1.loc[city], y1.loc[city], interpolate=False)
            endog = pd.concat([exog, endog.to_frame()], axis=1)
            model = StateSpaceModel(endog=endog, factors_x=5, factors_y=3)
            model.update(model.start_params)
            design, obs_cov, state_cov, transition = model.generate_start_matrices()
            self.assertTrue((transition == model.ssm['transition']).all().all(), msg='transition update error')
            self.assertTrue((design == model.ssm['design']).all().all(), msg='design update error')
            self.assertTrue((obs_cov == model.ssm['obs_cov']).all().all(), msg='obs_cov update error')
            self.assertTrue((state_cov == model.ssm['state_cov']).all().all(), msg='state_cov update error')
