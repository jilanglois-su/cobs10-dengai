import os
os.chdir('../')
import numpy as np
import pandas as pd
from src.d01_data.dengue_data_api import DengueDataApi
from src.d04_modeling.poisson_hmm import PoissonHMM
from src.d04_modeling.dynamic_factor_model import DynamicFactorModel
from src.d04_modeling.arx import ARX

dda = DengueDataApi()
x_train, x_validate, y_train, y_validate = dda.split_data(random=False)

cities = y_train.index.get_level_values('city').unique()

model_evaluation = pd.DataFrame(index=pd.Index([]), columns=['run_static', 'city_dummy', 'no_glm',
                                                             'num_states', 'city', 'lls', 'forecast_mae',
                                                             'in_mae', 'out_mae'])

for run_static in [True]:
    for city_dummy in [False]:
        for no_glm in [True, False]:
            for num_states in [2, 3, 4]:
                if no_glm:
                    if not run_static:
                        continue
                print(run_static, city_dummy, no_glm, num_states)
                num_components = 4 if run_static else 3

                results = {'run_static': run_static,
                           'city_dummy': city_dummy,
                           'no_glm': no_glm,
                           'num_states': num_states}
                if run_static:
                    num_components = 4
                    z_train, z_validate, pct_var = dda.get_pca(x_train.copy(), x_validate.copy(), num_components=num_components)
                    z_train['bias'] = 1.
                    z_validate['bias'] = 1.

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

                    z_train['bias'] = 1.
                    z_validate['bias'] = 1.

                if city_dummy:
                    z_validate['sj'] = 0.
                    z_validate.at[('sj', slice(None), slice(None)), 'sj'] = 1.
                    z_train['sj'] = 0.
                    z_train.at[('sj', slice(None), slice(None)), 'sj'] = 1.

                if no_glm:
                    columns2drop = z_train.columns[:num_components]
                    z_train.drop(columns=columns2drop, inplace=True)
                    z_validate.drop(columns=columns2drop, inplace=True)

                model = PoissonHMM(num_states=num_states, seed=1992)
                event_data_train = dict()
                event_data_train['x'] = model.format_event_data(z_train.droplevel('year'))
                event_data_train['y'] = model.format_event_data(y_train.droplevel('year'))

                lls_k, parameters_k = model.fit(event_data=event_data_train)

                event_data_validate = dict()
                event_data_validate['x'] = model.format_event_data(z_validate.droplevel('year'))
                event_data_validate['y'] = model.format_event_data(y_validate.droplevel('year').interpolate())

                y_viterbi_train, most_likely_states_train, lls_train = model.predict(event_data_train, parameters_k)
                y_viterbi_validate, most_likely_states_validate, lls_validate = model.predict(event_data_validate, parameters_k)
                forecasts, states_prob = model.forecast(event_data_train, event_data_validate, parameters_k, alpha=0.5)

                for i in range(len(event_data_train['y'])):
                    y_train_values = event_data_train['y'][i]
                    y_validate_values = event_data_validate['y'][i]
                    forecast_mae = np.abs(forecasts[i]['map']-y_validate_values).mean()
                    in_mae = np.abs(y_viterbi_train[i]-y_train_values).mean()
                    out_mae = np.abs(y_viterbi_validate[i]-y_validate_values).mean()

                    results['forecast_mae'] = forecast_mae
                    results['in_mae'] = in_mae
                    results['out_mae'] = out_mae
                    results['lls'] = lls_validate
                    results['city'] = cities[i]

                    model_evaluation = model_evaluation.append(results, ignore_index=True)

                print(model_evaluation.to_string())

model_evaluation.to_csv('model_evaluation_glm.csv')

