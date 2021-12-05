from unittest import TestCase
from src.d04_modeling.iohmm import IOHMM
from src.d01_data.dengue_data_api import DengueDataApi


class TestHmmEm(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("setUp")
        dda = DengueDataApi()
        x_train, x_validate, y_train, y_validate = dda.split_data()
        z_train, z_validate, pct_var = dda.get_pca(x_train, x_validate, num_components=4)
        cls.model = IOHMM(num_states=3, num_features=z_train.shape[1])
        cls.event_data = cls.model.format_event_data(z_train.droplevel('year'), y_train.droplevel('year'))

    def test_e_step(self):
        expectations, transition_expectations = self.model.e_step(self.event_data)
        for p in range(len(self.event_data)):
            num_periods = self.event_data[p][0].shape[0]
            test_value1 = expectations[p].sum().sum().item()
            test_value2 = transition_expectations[p].sum().sum().sum().item()
            self.assertAlmostEqual(num_periods, test_value1, places=0)
            self.assertAlmostEqual(num_periods, test_value2, places=0)

