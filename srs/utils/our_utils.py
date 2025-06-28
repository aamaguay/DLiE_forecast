import locale
import os
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import calendar
import torch
import random
from pathlib import Path
from typing import Tuple, Union, Dict, List
import optuna
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from srs.models.gam import forecast_gam, forecast_gam_whole_sample, forecast_gam_whole_sample_justTrainig
from srs.utils.tutor_utils import forecast_expert_ext, forecast_expert_ext_modifed
from srs.models.light_gbm import forecast_lgbm_whole_sample, forecast_lgbm_whole_sample_LongShortTerm_w_Optuna, forecast_lgbm_whole_sample_optuna_selectBestOptions, forecast_lgbm_whole_sample_justTrainig, forecast_lgbm_whole_sample_LongShortTerm_w_Optuna_justTrainig #forecast_lgbm_whole_sample_w_Optuna

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_holidays_dummy(dat, full_dates, holidays_ds, dayahead=1.5, daybefore=0.5):
    """
    Creates a holiday dummy variable expanded to hourly level.
    
    Args:
        dat (torch.Tensor): Input 3D tensor of shape (T, 24, V).
        full_dates (list or pd.Series): Daily datetime entries corresponding to dat (length = T).
        holidays_ds (pd.DataFrame): Holiday dataset with a 'Date' column.
        dayahead (float): How many days *after* a holiday to include as holiday effect.
        daybefore (float): How many days *before* a holiday to include as holiday effect.

    Returns:
        torch.Tensor: A (T*24, 1) dummy vector with 1 during holiday effect periods.
    """

    # Step 1: Prepare holiday dates
    holidays_ds["Date"] = pd.to_datetime(holidays_ds["Date"])
    holidays_filtered = holidays_ds[(holidays_ds["Date"].dt.year >= 2019) & 
                                    (holidays_ds["Date"].dt.year <= 2024) &
                                    (holidays_ds["CountryCode"] == "NO")]
    
    # Convert to datetime
    full_dates = pd.to_datetime(full_dates)
    T = dat.shape[0]
    S = dat.shape[1]
    full_datetime_index = pd.date_range(start=full_dates.iloc[0], periods=T*S, freq="H")
    dayahead = dayahead


    # Step 2: Create holiday dummy
    is_holiday = torch.zeros(T * S, dtype=torch.float32, device=dat.device)

    for holiday in holidays_filtered["Date"]:
        holiday_up =  holiday + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) 
        lower = holiday - pd.Timedelta(days=daybefore)
        upper = holiday_up + pd.Timedelta(days=dayahead)
        mask = (full_datetime_index >= lower) & (full_datetime_index <= upper)
        mask_indices = np.where(mask)[0]
        is_holiday[mask_indices] = 1.0
    
    return is_holiday.unsqueeze(1)

def prepare_data_forTraining(ds_weather_zone, dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags, weather_lags,
                             full_dates, holidays_ds, model_first_diff_price = False, dayahead=1.5, daybefore=0.5):
    def get_lagged(Z, lag):
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=device)
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])
    
    def get_lagged_firstDiff(Z, lag):
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=device)
        if lag == 0:
            return Z
        return torch.cat([
            torch.full((lag,), float('nan'), device=Z.device),
            Z[:-lag]
        ])

    def get_lagged_2d(Z, lag):
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=device)
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=device), Z[:-lag]], dim=0)
    
    def rolling_median(series, window):
        output = torch.full_like(series, float('nan'))
        for i in range(window, len(series)):
            output[i] = series[i - window:i].median()
        return output
    
    def pct_change(series):
        result = torch.full_like(series, float('nan'))
        result[1:] = (series[1:] - series[:-1]) / series[:-1] * 100
        return result
    
    def rolling_std(series, window):
        output = torch.full_like(series, float('nan'))
        for i in range(window, len(series)):
            output[i] = series[i - window:i].std()
        return output
    
    def second_diff(series):
        # 2nd difference: diff of first difference
        first_diff = series[1:] - series[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        second_diff = torch.cat([
            torch.tensor([float('nan'), float('nan')], device=series.device),
            second_diff
        ])
        return second_diff
    
    reg_names = list(reg_names) + ds_weather_zone.columns.tolist()
    # Make sure ds_weather_zone is sorted by time
    ds_weather_zone = ds_weather_zone.sort_index()

    # Convert DataFrame to numpy
    weather_array = ds_weather_zone.to_numpy()  # Shape: (days * 24, weather_features)

    # Reshape to 3D: (days, 24, weather_features)
    n_hours_per_day = 24
    n_days = weather_array.shape[0] // n_hours_per_day
    n_weather_features = weather_array.shape[1]

    weather_tensor = torch.tensor(
        weather_array.reshape(n_days, n_hours_per_day, n_weather_features),
        dtype=torch.float32,  # Or float64 if you use that
        device=device
    )

    S = dat.shape[1]  # 24 hours
    T = dat.shape[0]  # 731 days

    # --- get holidays ---
    holiday_hour_dummy = get_holidays_dummy(dat, full_dates, holidays_ds, dayahead, daybefore)


    # --- Prepare weekday dummies ---
    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)
    WD_full = WD.repeat_interleave(S, dim=0)

    # --- Indices ---
    reg_names = list(reg_names)
    price_idx = reg_names.index("Price")
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"] 
    da_forecast_names = [da_forecast_names[0], da_forecast_names[2]]
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)
    weather_vars = ["Temp", "Solar", "WindS", "WindDir", "Press", "Humid"]
    weather_idx = torch.tensor([reg_names.index(name) for name in weather_vars], device=device)

    # --- Flatten data ---
    flat_dat = dat.reshape(-1, dat.shape[2])  # shape (T*S, V)
    weather_tensor = weather_tensor.reshape(-1, weather_tensor.shape[2]) 
    flat_dat = torch.cat([flat_dat, weather_tensor], dim=1)
    price_series = flat_dat[:, price_idx]

    if model_first_diff_price:
        price_diff_series_ = price_series[1:] - price_series[:-1]
        price_diff_series_ = torch.cat([
            torch.tensor([float('nan')], device=price_diff_series_.device),
            price_diff_series_
        ])
        mat_price_lags = torch.stack([get_lagged_firstDiff(price_diff_series_, lag) for lag in price_s_lags], dim=1)
    else:
        mat_price_lags = torch.stack([get_lagged(price_series, lag) for lag in price_s_lags], dim=1)


    da_all = []
    for i in da_idx:
        series = flat_dat[:, i].detach().clone()
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    mat_weather_input = flat_dat[:, weather_idx]
    mat_weather = torch.cat([get_lagged_2d(mat_weather_input, lag) for lag in weather_lags], dim=1)

    # volatility variables
    price_tensor = flat_dat[:, price_idx]
    volatility_24h = rolling_std(price_tensor, window=24)   # 1-day volatility
    volatility_72h = rolling_std(price_tensor, window=72)  # 3-day volatility
    diff_pct = pct_change(price_tensor)  # Already in your code
    volatility_pct_24h = rolling_std(diff_pct, window=24)

    volatility_feats = torch.stack([
        get_lagged(volatility_24h, 1),
        get_lagged(volatility_pct_24h, 1)
        ], dim=1)
    
    # --- Hour-of-day cyclical encoding ---
    hour_indices = torch.arange(S, device=device).repeat(T)  # shape: (T*S,)
    hour_norm = 2 * torch.pi * hour_indices / 24

    sin_hour = torch.sin(hour_norm).unsqueeze(1)  # shape: (T*S, 1)
    cos_hour = torch.cos(hour_norm).unsqueeze(1)  # shape: (T*S, 1)

    total_hours = S * T  # Total hourly points

    # --- Weekly seasonality (168 hours cycle) ---
    hour_indices = torch.arange(total_hours, device=device)
    week_norm = 2 * torch.pi * hour_indices / (24 * 7)

    sin_week = torch.sin(week_norm).unsqueeze(1)
    cos_week = torch.cos(week_norm).unsqueeze(1)

    # --- Yearly seasonality (8760 hours) ---
    year_norm = 2 * torch.pi * hour_indices / (24 * 365)

    sin_year = torch.sin(year_norm).unsqueeze(1)
    cos_year = torch.cos(year_norm).unsqueeze(1)


    # here first difference
    if model_first_diff_price == True:
        price_series = price_series[1:] - price_series[:-1]
        price_series = torch.cat([
            torch.tensor([float('nan')], device=price_series.device),  # or use 0.0 instead of NaN if preferred
            price_series
            ])
    if isinstance(price_series, np.ndarray):
        price_series = torch.tensor(price_series, dtype=torch.float32, device=device)

    # additional variables
    # "Load_DA" -- pct_change, 
    # "Load_DA" -- lag(168),
    # "Price" -- rolling window, median 2 days
    load_da_tensor = flat_dat[:, reg_names.index("Load_DA")]
    roll2H_median_Price = rolling_median(price_tensor, window=2)
    pct_chg_Load_DA = pct_change(load_da_tensor)
    lag168_Load_DA = get_lagged(load_da_tensor, 168)
    volatility_load_DA_24h = rolling_std(load_da_tensor, window=24)
    volatility_load_DA_pct_24h = rolling_std(pct_chg_Load_DA, window=24)

    # Compute 2nd difference and its lag
    price_2nd_diff = second_diff(flat_dat[:, price_idx])
    lag1_price_2nd_diff = get_lagged(price_2nd_diff, 1)  # optional lag of 1

    extra_feats_tensor = torch.stack([
        pct_chg_Load_DA,
        lag168_Load_DA,
        lag1_price_2nd_diff
    ], dim=1)

    # --- Create dummy variable for hour 7 or 8 AM ---
    hour_indices = torch.arange(S, device=device).repeat(T)  # shape: (T*S,)
    hour_dummy_7to9_17to19 = ((hour_indices >= 7) & (hour_indices <= 9) | 
                              (hour_indices >= 17) & (hour_indices <= 19)).float().unsqueeze(1)  # shape: (T*S, 1)


    # join all data
    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, 
                    WD_full, mat_fuels, mat_weather,
                    extra_feats_tensor,
                    sin_hour, cos_hour,
                    sin_week, cos_week, 
                    sin_year, cos_year,
                    volatility_feats], dim=1)
    
    # --- feature names ----
    if model_first_diff_price == True:
        # 1. Price feature
        feature_names_Xy = ["deltaPrice"]

        # 2. Price lags
        feature_names_Xy += [f"deltaPrice_lag_{lag}" for lag in price_s_lags]
    else:
        # 1. Price feature
        feature_names_Xy = ["Price"]

        # 2. Price lags
        feature_names_Xy += [f"Price_lag_{lag}" for lag in price_s_lags]


    # 3. DA features
    da_feature_names = [reg_names[i] for i in da_idx.cpu().numpy()]
    for da_name in da_feature_names:
        for lag in da_lag:
            feature_names_Xy.append(f"{da_name}_lag_{lag}")

    # 4. Weekday dummy features
    feature_names_Xy += [f"WD_{day}" for day in wd]

    # 5. Fuel features
    fuel_feature_names = [reg_names[i] for i in fuel_idx.cpu().numpy()]
    for fuel_name in fuel_feature_names:
        for lag in fuel_lags:
            feature_names_Xy.append(f"{fuel_name}_lag_{lag}")
    
    # 6. Weather features
    weather_feature_names = [reg_names[i] for i in weather_idx.cpu().numpy()]
    for w_name in weather_feature_names:
        for lag in weather_lags:
            feature_names_Xy.append(f"{w_name}_lag_{lag}")

    feature_names_Xy += [
    "pct_chg_Load_DA",
    "lag168_Load_DA",
    "lag1_price_2nd_diff",
    "sin_hour", "cos_hour", 
    "sin_week", "cos_week",
    "sin_year", "cos_year",  
    "volatility_24h_lg1",
    "volatility_pct_24h_lg1"
    ]
    # --- ------ ----
    # Convert Xy to 3D: (days, hours, num_features)
    num_features = Xy.shape[1]
    Xy = Xy.reshape(T, S, num_features)
    return(feature_names_Xy, Xy)

def run_forecast_step_modified_JustTraining(
    full_price_original,
    full_price_first_lag,
    n,
    data_array,
    train_start_idx,
    train_end_idx,
    full_dates,
    feature_names, 
    ls_models,
    apply_smoo_spline_over_varList,
    n_trials_lgbm,
    days_for_st_model
):
    """
    n               : offset into the 2024 evaluation period
    train_start_idx : integer index of 2019-01-01 in dates_S
    train_end_idx   : integer index of 2023-12-31 in dates_S
    """
    if n == 0:
        reduction = 0
    else:
        reduction = 60 * ((n - 1) // 100 + 1)
    # compute the window bounds
    start_idx = train_start_idx + n + reduction # rolling window (train_start_idx + n)
    end_idx   = train_end_idx  +  n   # inclusive
    
    # slice out the training data & dates
    dat_slice = data_array[start_idx : (end_idx + 2)]
    days      = pd.Series(full_dates[start_idx : (end_idx + 2) ])  # <- here, if i use end_idx, it should be 2
    full_price_original     =  full_price_original[start_idx : (end_idx + 2)].reshape(-1)
    full_price_first_lag    =  full_price_first_lag[start_idx : (end_idx + 2)].reshape(-1)
    
    # # true price of the forecast day for evaluation
    # forecast_date_idx = end_idx + 1
    # true_price = price_S[forecast_date_idx]

    ls_results = []
    
    print(f"Loop {n:3d}: train {full_dates[start_idx]} -> {full_dates[end_idx]}, "
          f"forecast {full_dates[end_idx+1]}")
    print(f"  dat_slice shape: {dat_slice.shape}  → flatten count = {dat_slice.shape[0]*24}")
    
    for md in ls_models:
        if md == "lgbm_24hAhead_defaultHyper":
            # lg_gbm model forecast, 24h ahead
            lg_gbm_forecast = forecast_lgbm_whole_sample_justTrainig(
            dat_slice,
            feature_names,
            full_price_original,
            full_price_first_lag
            )["forecasts"]
            ls_results.append(lg_gbm_forecast)
        
        if md == "gam_24hAhead":
            # gam model forecast, 24h ahead
            lg_gam_forecast = forecast_gam_whole_sample_justTrainig(
            dat_slice,
            feature_names, 
            apply_smoo_spline_over_varList
            )["forecasts"]
            ls_results.append(lg_gam_forecast)

        if md == "lgbm_24hAhead_withOptune":
            # lg_gbm model forecast, 24h ahead
            lg_gbm_forecast_wOpt = forecast_lgbm_whole_sample_w_Optuna(
            dat_slice,
            feature_names, 
            n_trials_lgbm
            )["forecasts"]
            ls_results.append(lg_gbm_forecast_wOpt)

        if md == "lgbm_24hAhead_LongShortTermWithOptune":
            # lg_gbm model forecast, 24h ahead
            lg_gbm_forecast_wOpt = forecast_lgbm_whole_sample_LongShortTerm_w_Optuna_justTrainig(
            dat_slice,
            feature_names, 
            n_trials_lgbm,
            days_for_st_model
            )["forecasts"]
            ls_results.append(lg_gbm_forecast_wOpt)

    print(f"-> finished loop {n}")
    return n, ls_results

def run_forecast_step_modified(
    n,
    price_S,
    data_array,
    train_start_idx,
    train_end_idx,
    full_dates,
    wd,
    price_s_lags,
    da_lag,
    feature_names,n_trials_lgbm,
    days_for_st_model
):
    """
    n               : offset into the 2024 evaluation period
    train_start_idx : integer index of 2019-01-01 in dates_S
    train_end_idx   : integer index of 2023-12-31 in dates_S
    """
    if n == 0:
        reduction = 0
    else:
        reduction = 60 * ((n - 1) // 100 + 1)
    # compute the window bounds
    start_idx = train_start_idx + n + reduction # rolling window (train_start_idx + n)
    end_idx   = train_end_idx  +  n   # inclusive
    
    # slice out the training data & dates
    dat_slice = data_array[start_idx : (end_idx + 2)]
    days      = pd.Series(full_dates[start_idx : (end_idx + 2) ])  # <- here, if i use end_idx, it should be 2
    
    # # true price of the forecast day for evaluation
    # forecast_date_idx = end_idx + 1
    # true_price = price_S[forecast_date_idx]
    
    print(f"Loop {n:3d}: train {full_dates[start_idx]} -> {full_dates[end_idx]}, "
          f"forecast {full_dates[end_idx+1]}")
    print(f"  dat_slice shape: {dat_slice.shape}  → flatten count = {dat_slice.shape[0]*24}")
    
    # GAM forecast (24 h ahead)
    # gam_forecast_24h = forecast_gam_whole_sample(
    #     dat=dat_slice,
    #     days=days,
    #     wd=wd,
    #     price_s_lags=price_s_lags,
    #     da_lag=da_lag,
    #     reg_names=feature_names,
    #     fuel_lags=[2]
    # )["forecasts"]
    
    # # per‐hour GAM forecast
    # gam_forecast_per_hour = forecast_gam(
    #     dat=dat_slice,
    #     days=days,
    #     wd=wd,
    #     price_s_lags=price_s_lags,
    #     da_lag=da_lag,
    #     reg_names=feature_names,
    #     fuel_lags=[2]
    # )["forecasts"]
  
# # Expert model forecast
#   expert_forecast = forecast_expert_ext_modifed(
#       dat=dat_slice,
#       days=days,
#       wd=wd,
#       price_s_lags=price_s_lags,
#       da_lag=da_lag,
#       reg_names=feature_names,
#       fuel_lags=[2]
#   )["forecasts"]

    # # lg_gbm model forecast
    # lg_gbm_forecast = forecast_lgbm_whole_sample(
    #   dat=dat_slice,
    #   days=days,
    #   wd=wd,
    #   price_s_lags=price_s_lags,
    #   da_lag=da_lag,
    #   reg_names=feature_names,
    #   fuel_lags=[2]
    # )["forecasts"]

    # lg_gbm model forecast
    lg_gbm_forecast = forecast_lgbm_whole_sample_LongShortTerm_w_Optuna(
      dat=dat_slice,
      days=days,
      wd=wd,
      price_s_lags=price_s_lags,
      da_lag=da_lag,
      reg_names=feature_names,
      fuel_lags=[2],
      n_trials_lgbm = n_trials_lgbm,
      days_for_st_model = days_for_st_model
    )["forecasts"]
    
    print(f"-> finished loop {n}")
    return n, lg_gbm_forecast

def run_forecast_step(
    n,
    price_S,
    data_array,
    train_start_idx,
    train_end_idx,
    full_dates,
    wd,
    price_s_lags,
    da_lag,
    feature_names,
):
    """
    n               : offset into the 2024 evaluation period
    train_start_idx : integer index of 2019-01-01 in dates_S
    train_end_idx   : integer index of 2023-12-31 in dates_S
    """
    # compute the window bounds
    start_idx = train_start_idx
    end_idx   = train_end_idx + n   # inclusive
    
    # slice out the training data & dates
    dat_slice = data_array[start_idx : end_idx + 1]
    days      = pd.Series(full_dates[start_idx : end_idx + 1])  # <- Change to_datetime to Series, otherwise \
                                                                # weekdays_num inside gam_forecast_24h = forecast_gam_whole_sample()
                                                                # will crash

    
    # # true price of the forecast day for evaluation
    # forecast_date_idx = end_idx + 1
    # true_price = price_S[forecast_date_idx]
    
    print(f"Loop {n:3d}: train {full_dates[start_idx]} -> {full_dates[end_idx]}, "
          f"forecast {full_dates[end_idx+1]}")
    print(f"  dat_slice shape: {dat_slice.shape}  → flatten count = {dat_slice.shape[0]*24}")
    
    # GAM forecast (24 h ahead)
    gam_forecast_24h = forecast_gam_whole_sample(
        dat=dat_slice,
        days=days,
        wd=wd,
        price_s_lags=price_s_lags,
        da_lag=da_lag,
        reg_names=feature_names,
        fuel_lags=[2]
    )["forecasts"]
    
    # # per‐hour GAM forecast
    # gam_forecast_per_hour = forecast_gam(
    #     dat=dat_slice,
    #     days=days,
    #     wd=wd,
    #     price_s_lags=price_s_lags,
    #     da_lag=da_lag,
    #     reg_names=feature_names,
    #     fuel_lags=[2]
    # )["forecasts"]
  
# # Expert model forecast
#   expert_forecast = forecast_expert_ext_modifed(
#       dat=dat_slice,
#       days=days,
#       wd=wd,
#       price_s_lags=price_s_lags,
#       da_lag=da_lag,
#       reg_names=feature_names,
#       fuel_lags=[2]
#   )["forecasts"]

#   # lg_gbm model forecast
#   lg_gbm_forecast = forecast_lgbm_whole_sample(
#       dat=dat_slice,
#       days=days,
#       wd=wd,
#       price_s_lags=price_s_lags,
#       da_lag=da_lag,
#       reg_names=feature_names,
#       fuel_lags=[2]
#   )["forecasts"]
    
    print(f"-> finished loop {n}")
    return n, gam_forecast_24h