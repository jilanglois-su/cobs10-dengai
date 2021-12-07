from unittest import TestCase
import numpy as np
from src.d01_data.dengue_data_api import DengueDataApi
from src.d04_modeling.poisson_glm import PoissonGLM
from expected_values import *


class TestPoissonGlm(TestCase):

    def test_loglikeihood(self):
        dda = DengueDataApi()
        x_train, x_validate, y_train, y_validate = dda.split_data()
        sigma2 = 1.
        poisson_glm = PoissonGLM(x_train=x_train, y_train=y_train, sigma2=sigma2)
        weights = np.ones(len(x_train)).reshape((-1, 1))
        obj = lambda w: -poisson_glm.log_joint(y_train.values.reshape((-1, 1)), x_train.values, weights, w, sigma2) / len(x_train)
        obj_grad = lambda w: -poisson_glm.log_joint_grad(y_train.values.reshape((-1, 1)), x_train.values, weights, w, sigma2) / len(x_train)
        obj_hess = lambda w: -poisson_glm.log_joint_hess(y_train.values.reshape((-1, 1)), x_train.values, weights, w, sigma2) / len(x_train)

        w = 0.5 * np.ones(x_train.shape[1])

        self.assertIsInstance(obj(w), np.float64, msg="log joint not float")
        self.assertAlmostEqual(first=expected_obj_log_joint, second=obj(w), places=6,
                               msg="Error Poisson GLM Log-joint")
        self.assertEqual(obj_grad(w).shape, w.shape, msg="log joint grad not right size")
        self.assertMultiLineEqual(first=np.array2string(expected_obj_log_joint_grad, precision=5), second=np.array2string(obj_grad(w), precision=5),
                                  msg="Error Poisson GLM Log-joint Grad")
        self.assertEqual(obj_hess(w).shape, (w.shape[0], w.shape[0]), msg="log joint hess not right size")
        self.assertMultiLineEqual(first=np.array2string(expected_obj_log_joint_hess, precision=5), second=np.array2string(obj_hess(w), precision=5),
                                  msg="Error Poisson GLM Log-joint Hessian")

    def test_model(self):
        dda = DengueDataApi()
        x_train, x_validate, y_train, y_validate = dda.split_data()
        sigma2 = 1.
        poisson_glm = PoissonGLM(x_train=x_train, y_train=y_train, sigma2=sigma2)

        poisson_glm.compute_posterior_mode()
        poisson_glm.compute_laplace_approximation()

        log_joint, mae = poisson_glm.validate_model(x_validate=x_validate, y_validate=y_validate,
                                                    show_posterior_predictive_dist=False)

        self.assertAlmostEqual(first=expected_log_joint, second=log_joint, places=11, msg="Error Poisson GLM Log-joint")
        self.assertAlmostEqual(first=expected_mae, second=mae, places=11, msg="Error Poisson GLM MAE")
