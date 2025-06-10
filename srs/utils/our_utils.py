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

from srs.models.gam import forecast_gam, forecast_gam_whole_sample
from srs.utils.tutor_utils import forecast_expert_ext, forecast_expert_ext_modifed
from srs.models.light_gbm import forecast_lgbm_whole_sample

def run_forecast_step(
    n,
    price_S,
    data_array,
    train_start_idx,
    train_end_idx,
    dates_S,
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
    days      = pd.to_datetime(dates_S[start_idx : end_idx + 1])
    
    # # true price of the forecast day for evaluation
    # forecast_date_idx = end_idx + 1
    # true_price = price_S[forecast_date_idx]
    
    print(f"Loop {n:3d}: train {dates_S[start_idx]} -> {dates_S[end_idx]}, "
          f"forecast {dates_S[end_idx+1]}")
    
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