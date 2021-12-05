PATH_DATA_RAW = "../data/01_raw/"
PATH_OUTPUT_NETWORK = "output_network_{name:}.pkl"
PATH_STATE_NETWORK = "state_network_{name:}.pkl"

NDVI_COLS = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
PERSIANN_COLS = ['precipitation_amt_mm']
NOAA_NCEP_COLS = ['reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
                  'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k',
                  'reanalysis_min_air_temp_k', 'reanalysis_precip_amt_kg_per_m2',
                  'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
                  'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k']
NOAA_GHCN_COLS = ['station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
                  'station_min_temp_c', 'station_precip_mm']

LOG_TRANSFORM = ['reanalysis_precip_amt_kg_per_m2']

FEATURE_COLS = NDVI_COLS + NOAA_NCEP_COLS
WEEK_START_DATE_COL = 'week_start_date'
INDEX_COLS = ['city', 'year', 'weekofyear']

diff_variables = {'ndvi_ne': True,
                  'ndvi_nw': True,
                  'ndvi_se': True,
                  'ndvi_sw': False,
                  'reanalysis_air_temp_k': False,
                  'reanalysis_avg_temp_k': False,
                  'reanalysis_dew_point_temp_k': False,
                  'reanalysis_max_air_temp_k': False,
                  'reanalysis_min_air_temp_k': False,
                  'reanalysis_precip_amt_kg_per_m2': True,
                  'reanalysis_relative_humidity_percent': False,
                  'reanalysis_sat_precip_amt_mm': False,
                  'reanalysis_specific_humidity_g_per_kg': False,
                  'reanalysis_tdtr_k': True}