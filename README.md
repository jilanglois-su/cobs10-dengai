# DengAI: Predicting Disease Spread, competition hosted by DRIVENDATA

We implemented a Input-Output Hidden Markov Model using the data from the 'DengAI: Predicting Disease Spread' competition hosted by DRIVENDATA. The main results are summarized in the report [Final_Report.pdf](https://github.com/jilanglois-su/cobs10-dengai/blob/8430415a371816a98aca12ff875325198fb8a4cb/Final_Report.pdf).

https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/82/

# Problem description

Your goal is to predict the `total_cases` label for each `(city, year, weekofyear)` in the test set. There are two cities, San Juan and Iquitos, with test data for each city spanning 5 and 3 years respectively. You will make one submission that contains predictions for both cities. The data for each `city` have been concatenated along with a city column indicating the source: `sj` for San Juan and `iq` for Iquitos. The test set is a pure future hold-out, meaning the test data are sequential and non-overlapping with any of the training data. Throughout, missing values have been filled as `NaN`s.

## The features in this dataset
You are provided the following set of information on a `(year, weekofyear)` timescale:

(Where appropriate, units are provided as a `_unit` suffix on the feature name.)

## City and date indicators
- `city` – City abbreviations: sj for San Juan and iq for Iquitos
- `week_start_date` – Date given in yyyy-mm-dd format
## NOAA's GHCN daily climate data weather station measurements
- `station_max_temp_c` – Maximum temperature
- `station_min_temp_c` – Minimum temperature
- `station_avg_temp_c` – Average temperature
- `station_precip_mm` – Total precipitation
- `station_diur_temp_rng_c` – Diurnal temperature range
## PERSIANN satellite precipitation measurements (0.25x0.25 degree scale)
- `precipitation_amt_mm` – Total precipitation
## NOAA's NCEP Climate Forecast System Reanalysis measurements (0.5x0.5 degree scale)
- `reanalysis_sat_precip_amt_mm` – Total precipitation
- `reanalysis_dew_point_temp_k` – Mean dew point temperature
- `reanalysis_air_temp_k` – Mean air temperature
- `reanalysis_relative_humidity_percent` – Mean relative humidity
- `reanalysis_specific_humidity_g_per_kg` – Mean specific humidity
- `reanalysis_precip_amt_kg_per_m2` – Total precipitation
- `reanalysis_max_air_temp_k` – Maximum air temperature
- `reanalysis_min_air_temp_k` – Minimum air temperature
- `reanalysis_avg_temp_k` – Average air temperature
- `reanalysis_tdtr_k` – Diurnal temperature range
## Satellite vegetation - Normalized difference vegetation index (NDVI) - NOAA's CDR Normalized Difference Vegetation Index (0.5x0.5 degree scale) measurements
- `ndvi_se` – Pixel southeast of city centroid
- `ndvi_sw` – Pixel southwest of city centroid
- `ndvi_ne` – Pixel northeast of city centroid
- `ndvi_nw` – Pixel northwest of city centroid
