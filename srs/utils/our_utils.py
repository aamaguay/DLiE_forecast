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

def run_forecast_step(n, 
                      price_S, 
                      data_array, 
                      begin_eval, 
                      D, 
                      dates_S, 
                      wd, 
                      price_s_lags, 
                      da_lag, 
                      reg_names, 
                      data_columns):
    
    # Save true price
    true_price = price_S[begin_eval + n]
    
    # Get days and data slice
    dat_slice = data_array[(begin_eval - D + n):(begin_eval + 1 + n)]
    print(f"START NS: {n}  index: {begin_eval + n}  data shape: {data_array.shape}")
    
    days = pd.to_datetime(dates_S[(begin_eval - D + n):(begin_eval + 1 + n)])

    # GAM forecast
    gam_forecast_24h = forecast_gam_whole_sample(
        dat=dat_slice,
        days=days,
        wd=wd,
        price_s_lags=price_s_lags,
        da_lag=da_lag,
        reg_names=data_columns,
        fuel_lags=[2]
    )["forecasts"]
    
        # GAM forecast
    gam_forecast_per_hour = forecast_gam(
        dat=dat_slice,
        days=days,
        wd=wd,
        price_s_lags=price_s_lags,
        da_lag=da_lag,
        reg_names=data_columns,
        fuel_lags=[2]
    )["forecasts"]

    # # Expert model forecast
    # expert_forecast = forecast_expert_ext_modifed(
    #     dat=dat_slice,
    #     days=days,
    #     wd=wd,
    #     price_s_lags=price_s_lags,
    #     da_lag=da_lag,
    #     reg_names=data_columns,
    #     fuel_lags=[2]
    # )["forecasts"]

    # # lg_gbm model forecast
    # lg_gbm_forecast = forecast_lgbm_whole_sample(
    #     dat=dat_slice,
    #     days=days,
    #     wd=wd,
    #     price_s_lags=price_s_lags,
    #     da_lag=da_lag,
    #     reg_names=data_columns,
    #     fuel_lags=[2]
    # )["forecasts"]

    print(f"END NS: {n}")
    return n, gam_forecast_24h, gam_forecast_per_hour  