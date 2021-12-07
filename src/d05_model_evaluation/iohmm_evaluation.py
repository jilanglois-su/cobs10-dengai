import os
os.chdir('../')
import numpy as np
import pandas as pd
from src.d01_data.dengue_data_api import DengueDataApi
from src.d04_modeling.iohmm import IOHMM
from src.d04_modeling.dynamic_factor_model import DynamicFactorModel
from src.d04_modeling.arx import ARX

dda = DengueDataApi()
x_train, x_validate, y_train, y_validate = dda.split_data(random=False)

cities = y_train.index.get_level_values('city').unique()

model_evaluation = pd.DataFrame(index=pd.Index([]), columns=['run_static', 'city_dummy', 'no_glm',
                                                             'num_states', 'city', 'lls', 'forecast_mae',
                                                             'in_mae', 'out_mae'])

for run_static in [True]:
    for num_states in [2, 3, 4]:
        print(run_static, num_states)
        num_components = 4 if run_static else 3

        results = {'run_static': run_static,
                   'num_states': num_states}
        if run_static:
            num_components = 4
            z_train, z_validate, pct_var = dda.get_pca(x_train.copy(), x_validate.copy(), num_components=num_components)

        else:
            dfm_model = DynamicFactorModel(x_train.copy(), y_train.copy(), factors=num_components, factor_orders=1, idiosyncratic_ar1=True)
            dfm_model.fit()

            z_train, y_train = dfm_model.get_filtered_factors(x_train.copy(), y_train.copy())
            z_validate, y_validate = dfm_model.get_filtered_factors(x_validate.copy(), y_validate.copy())

            arx_model = ARX(x_train=z_train.copy(), y_train=y_train.copy(), p={'iq': 2, 'sj': 3}, d=1)
            arx_model.fit()
            y_train_hat = []
            y_validate_hat = []
            for city in arx_model.get_cities():
                # print(city)
                y_train_hat += [arx_model.predict(city, z_train.copy())]
                y_validate_hat += [arx_model.predict(city, z_validate.copy())]
            y_train_hat = pd.concat(y_train_hat, axis=0, ignore_index=True)
            y_validate_hat = pd.concat(y_validate_hat, axis=0, ignore_index=True)
            mask = np.isnan(y_train).values
            y_train.loc[mask] = y_train_hat.loc[mask].round().values
            mask = np.isnan(y_validate).values
            y_validate.loc[mask] = y_validate_hat.loc[mask].round().values

        try:
            model = IOHMM(num_states=num_states, num_features=z_train.shape[1], load=True)
        except:
            model = IOHMM(num_states=num_states, num_features=z_train.shape[1], load=False)

        event_data_train = model.format_event_data(z_train.droplevel('year'), y_train.droplevel('year'))
        event_data_test = model.format_event_data(z_validate.droplevel('year'), y_validate.droplevel('year'))
        lls = model.fit(event_data=event_data_train, save=True, num_iterations=400)

        train_viterbi, _, _, _, _ = model.predict(event_data_train)
        test_viterbi, _, _, _, marginal_ll = model.predict(event_data_test)
        forecasts, states_prob = model.forecast(event_data_train, event_data_test)

        for p in range(len(event_data_test)):
            input_obs_train, output_obs_train = event_data_train[p]
            input_obs_test, output_obs_test = event_data_test[p]

            y_train_values = output_obs_train.detach().numpy().reshape((-1,))
            y_validate_values = output_obs_test.detach().numpy().reshape((-1,))
            forecast_mae = np.abs(forecasts[p]['map'] - y_validate_values).mean()
            in_mae = np.abs(train_viterbi[p].reshape((-1,)) - y_train_values).mean()
            out_mae = np.abs(test_viterbi[p].reshape((-1,)) - y_validate_values).mean()

            results['forecast_mae'] = forecast_mae
            results['in_mae'] = in_mae
            results['out_mae'] = out_mae
            results['lls'] = marginal_ll
            results['city'] = cities[p]

            model_evaluation = model_evaluation.append(results, ignore_index=True)

        print(model_evaluation.to_string())

model_evaluation.to_csv('model_evaluation_iohmm.csv')

