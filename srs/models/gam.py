
# load packages
import pandas as pd
import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from calendar import day_abbr
import torch
from pygam import LinearGAM, s, l
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forecast_gam(dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
    S = dat.shape[1]  # number of hours in a day

    print(f"shape of initial data: {dat.shape}")
    print(f"missing values of initial data: {np.isnan(dat).sum(axis=(0, 1))}")

    forecast = torch.full((S,), float('nan'), device=device)

    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)

    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    reg_names = list(reg_names)
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    price_idx = reg_names.index("Price")
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    def get_lagged(Z, lag):
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])

    def get_lagged_2d(Z, lag):
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=Z.device), Z[:-lag]], dim=0)

    mat_fuels = torch.cat([get_lagged_2d(dat[:, 0, fuel_idx], lag=l) for l in fuel_lags], dim=1)
    price_last = get_lagged(dat[:, S - 1, price_idx], lag=1)

    coefs = []

    for s_ in range(S):
        acty = dat[:, s_, price_idx]
        mat_price_lags = torch.stack([get_lagged(acty, lag) for lag in price_s_lags], dim=1)
        mat_da_forecasts = dat[:, s_, da_idx]

        da_lagged_list = [
            torch.stack([get_lagged(mat_da_forecasts[:, i], lag) for lag in da_lag], dim=1)
            for i in range(len(da_forecast_names))
        ]
        da_all_var = torch.cat(da_lagged_list, dim=1)

        if s_ == S - 1:
            regmat = torch.cat([acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels], dim=1)
        else:
            regmat = torch.cat([acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels, price_last.unsqueeze(1)], dim=1)

        print(regmat.shape)
        print("Missing values per column before we remove:", torch.isnan(regmat).sum(dim=0))

        nan_mask = ~torch.any(torch.isnan(regmat), dim=1)
        regmat0 = regmat[nan_mask]

        Xy = regmat0
        mean = Xy[:-1].mean(dim=0)
        std = Xy[:-1].std(dim=0)
        std[std == 0] = 1
        Xy_scaled = (Xy - mean) / std

        X = Xy_scaled[:-1, 1:].cpu().numpy()
        y = Xy_scaled[:-1, 0].cpu().numpy()
        x_pred = Xy_scaled[-1, 1:].cpu().numpy()

        n_features = X.shape[1]
        terms = s(0)
        for i in range(1, n_features):
            terms += s(i)
        model = LinearGAM(terms).fit(X, y)

        print('GAM fitted for hour', s_)

        y_pred = model.predict(np.array([x_pred]))[0]
        forecast[s_] = y_pred * std[0].item() + mean[0].item()

        coef = torch.tensor(model.coef_, dtype=torch.float32, device=device)
        coef[coef != coef] = 0
        coefs.append(coef.cpu().numpy())

    # Padding coefficient rows to the same length (GAMs may vary)
    max_len = max(len(row) for row in coefs)
    coefs_array = np.array([np.pad(row, (0, max_len - len(row)), constant_values=np.nan) for row in coefs])
    coefs_df = pd.DataFrame(coefs_array)

    return {"forecasts": forecast, "coefficients": coefs_df}


def forecast_gam_whole_sample(dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
    def get_lagged(Z, lag):
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag,), float('nan'), device=device), Z[:-lag]])

    def get_lagged_2d(Z, lag):
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
        series = flat_dat[:, i]
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, WD_full, mat_fuels], dim=1)

    # --- Split into training and prediction sets ---
    # y = price_series
    mask = ~torch.isnan(Xy).any(dim=1)
    Xy = Xy[mask]
    # y = y[mask]

    # Separate last 24 hours (forecast target)
    n_total = Xy.shape[0]
    last_day_indices = torch.arange(n_total - S, n_total, device=device)

    # estimate mean and std dv
    mean = Xy[:-S,:].mean(dim=0)
    std = Xy[:-S,:].std(dim=0)
    std[std == 0] = 1 # Prevent division by zero
    Xy_scaled = (Xy - mean) / std
          

    # Separate last 24 hours (forecast target)
    forecast_x = Xy_scaled[last_day_indices,1:]
    train_mask = torch.ones(n_total, dtype=torch.bool, device=device)
    train_mask[last_day_indices] = False

    X_train = Xy_scaled[train_mask, 1:].cpu().numpy()
    y_train = Xy_scaled[train_mask,0].cpu().numpy()
    x_pred = forecast_x.cpu().numpy()

    # Convert to Python lists and concatenate
    idx_fuel_da = fuel_idx.tolist() + da_idx.tolist()

    # --- Fit GAM ---
    terms = s(idx_fuel_da[0])
    for i in idx_fuel_da[1:]:
        terms += s(i)
    
    gam = LinearGAM(terms).fit(X_train, y_train)
    # print(gam)

    # --- Predict and collect output ---
    y_pred = gam.predict(x_pred)

    y_pred = ( (y_pred) * std[0].cpu().item() ) + mean[0].cpu().item()
    # print(y_pred)

    # --- Save model coefficients ---
    # Extract all coefficients and the mapping to features
    coef = gam.coef_
    term_features = gam.terms.feature  # array mapping each coef to a feature index (or None for intercept)

    # Group coefficients by feature
    coef_dict = defaultdict(list)
    for c, f in zip(coef, term_features):
        if f is not None:
            coef_dict[f].append(c)

    # Pad rows to same length
    max_len = max(len(v) for v in coef_dict.values())
    coef_matrix = np.array([
        np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
        for k, v in sorted(coef_dict.items())
    ])

    # Assign feature names
    feature_names = ['Feature_' + str(i) for i in sorted(coef_dict.keys())]
    col_names = [f'coef_{j}' for j in range(max_len)]
    coefs_df = pd.DataFrame(coef_matrix, index=feature_names, columns=col_names)

    return {
    "coef_df": coefs_df,
    "statistics_": gam.statistics_,
    "n_features": X_train.shape[1],
    "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device)
    }

def forecast_gam_whole_sample_justTrainig(Xy, feature_names_Xy, apply_spline_over_varList):
    def build_gam_terms(feature_names, apply_spline_over_varList):
        terms = None
        for idx, name in enumerate(feature_names):
            term = s(idx) if name in apply_spline_over_varList else l(idx)
            terms = term if terms is None else terms + term
        return terms
    
    S = Xy.shape[1]  # 24 hours
    T = Xy.shape[0]  # 731 days
    Xy = Xy.reshape(-1, Xy.shape[-1]) 
    # --- Split into training and prediction sets ---
    mask = ~torch.isnan(Xy).any(dim=1)
    Xy = Xy[mask]

    # Separate last 24 hours (forecast target)
    n_total = Xy.shape[0]
    last_day_indices = torch.arange(n_total - S, n_total, device=device)

    # estimate mean and std dv
    mean = Xy[:-S,:].mean(dim=0)
    std = Xy[:-S,:].std(dim=0)
    std[std == 0] = 1 # Prevent division by zero
    Xy_scaled = (Xy - mean) / std
          

    # Separate last 24 hours (forecast target)
    forecast_x = Xy_scaled[last_day_indices,1:]
    train_mask = torch.ones(n_total, dtype=torch.bool, device=device)
    train_mask[last_day_indices] = False

    X_train = Xy_scaled[train_mask, 1:].cpu().numpy()
    y_train = Xy_scaled[train_mask,0].cpu().numpy()
    x_pred = forecast_x.cpu().numpy()

    # create s-terms
    terms = build_gam_terms(feature_names_Xy[1:], apply_spline_over_varList)
    
    # ft a gam
    gam = LinearGAM(terms).fit(X_train, y_train)

    # --- Predict and collect output ---
    y_pred = gam.predict(x_pred)

    y_pred = ( (y_pred) * std[0].cpu().item() ) + mean[0].cpu().item()

    coef = gam.coef_
    term_features = gam.terms.feature  # array mapping each coef to a feature index (or None for intercept)

    return {
    "statistics_": gam.statistics_,
    "n_features": X_train.shape[1],
    "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device)
    }