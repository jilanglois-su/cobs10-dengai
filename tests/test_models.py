from unittest import TestCase
import os
import seaborn as sns
import pandas as pd
from src.d01_data.dengue_data_api import DengueDataApi
from src.d04_modeling.poisson_glm import PoissonGLM


class TestModels(TestCase):

    def test_poisson_glm(self):
        expected_log_joint = 35007.52445104625
        expected_mae = 27.09514170040486
        dda = DengueDataApi()
        x_train, x_validate, y_train, y_validate = dda.split_data()
        sigma2 = 1.
        poisson_glm = PoissonGLM(x_train=x_train, y_train=y_train, sigma2=sigma2)

        _, w_hist = poisson_glm.compute_posterior_mode()
        poisson_glm.compute_laplace_approximation()
        cov_map = poisson_glm.get_cov_map()

        log_joint, mae = poisson_glm.validate_model(x_validate=x_validate, y_validate=y_validate,
                                                    show_posterior_predictive_dist=False)

        self.assertAlmostEqual(first=expected_log_joint, second=log_joint, places=11, msg="Error Poisson GLM Log-joint")
        self.assertAlmostEqual(first=expected_mae, second=mae, places=11, msg="Error Poisson GLM MAE")
