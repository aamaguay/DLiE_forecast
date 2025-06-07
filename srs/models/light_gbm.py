# load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import torch
import lightgbm as lgb
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def forecast_lgbm_whole_sample(dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
    def get_lagged(Z, lag):
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=device)
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])

    def get_lagged_2d(Z, lag):
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=device)
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=device), Z[:-lag]], dim=0)



    S = dat.shape[1]  # 24 hours
    T = dat.shape[0]  # 731 days

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
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    # --- Flatten data ---
    flat_dat = dat.reshape(-1, dat.shape[2])  # shape (T*S, V)
    price_series = flat_dat[:, price_idx]

    # --- Feature Engineering ---
    mat_price_lags = torch.stack([get_lagged(price_series, lag) for lag in price_s_lags], dim=1)

    da_all = []
    for i in da_idx:
        series = torch.tensor(flat_dat[:, i], dtype=torch.float32, device=device)
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    price_series = flat_dat[:, price_idx]
    if isinstance(price_series, np.ndarray):
        price_series = torch.tensor(price_series, dtype=torch.float32, device=device)

    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, WD_full, mat_fuels], dim=1)

    # --- Clean NaNs ---
    mask = ~torch.isnan(Xy).any(dim=1)
    Xy = Xy[mask]

    # --- Split into train and forecast sets ---
    n_total = Xy.shape[0]
    last_day_indices = torch.arange(n_total - S, n_total, device=device)

    # Normalize data
    mean = Xy[:-S, :].mean(dim=0)
    std = Xy[:-S, :].std(dim=0)
    std[std == 0] = 1
    Xy_scaled = (Xy - mean) / std

    forecast_x = Xy_scaled[last_day_indices, 1:]
    train_mask = torch.ones(n_total, dtype=torch.bool, device=device)
    train_mask[last_day_indices] = False

    X_train = Xy_scaled[train_mask, 1:].cpu().numpy()
    y_train = Xy_scaled[train_mask, 0].cpu().numpy()
    x_pred = forecast_x.cpu().numpy()

    # --- Train LightGBM ---
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
    lgb_model.fit(X_train, y_train)

    # --- Predict and denormalize ---
    y_pred = lgb_model.predict(x_pred)
    y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()

    return {
        "model": lgb_model,
        "n_features": X_train.shape[1],
        "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device)
    }
