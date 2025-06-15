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
import optuna

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
    
    def rolling_median(series, window):
        output = torch.full_like(series, float('nan'))
        for i in range(window, len(series)):
            output[i] = series[i - window:i].median()
        return output
    
    def pct_change(series):
        result = torch.full_like(series, float('nan'))
        result[1:] = (series[1:] - series[:-1]) / series[:-1] * 100
        return result



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
        series = flat_dat[:, i].detach().clone()
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    price_series = flat_dat[:, price_idx]
    if isinstance(price_series, np.ndarray):
        price_series = torch.tensor(price_series, dtype=torch.float32, device=device)

    # additional variables
    # "Load_DA" -- pct_change, 
    # "Load_DA" -- lag(168),
    # "Price" -- rolling window, median 2 days
    price_tensor = flat_dat[:, reg_names.index("Price")]
    load_da_tensor = flat_dat[:, reg_names.index("Load_DA")]
    roll2d_median_Price = rolling_median(price_tensor, window=48)
    pct_chg_Load_DA = pct_change(load_da_tensor)
    lag168_Load_DA = get_lagged(load_da_tensor, 168)

    extra_feats_tensor = torch.stack([
        pct_chg_Load_DA,
        lag168_Load_DA,
        roll2d_median_Price
    ], dim=1)

    # join all data
    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, WD_full, mat_fuels, extra_feats_tensor], dim=1)

    # --- feature names ----
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

    feature_names_Xy += ["pct_chg_Load_DA", "lag168_Load_DA", "roll2d_median_Price"]
    # --- ------ ----

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
    x_pred = pd.DataFrame(x_pred, columns=feature_names_Xy[1:])

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

# early stopping class
class StopAfterBestStalls:
    def __init__(self, patience: int = 3, min_gain: float = 0.001):
        self.patience = patience
        self.min_gain = min_gain
        self.best_value = None
        self.counter = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        current_value = trial.value

        if self.best_value is None or current_value < self.best_value - self.min_gain:
            self.best_value = current_value
            self.counter = 0
            print(f"[Trial {trial.number}] New best: {current_value:.5f}")
        else:
            self.counter += 1
            print(f"[Trial {trial.number}] No significant gain (Δ ≤ {self.min_gain}). Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"\n⏹️  Early stop: no gain > {self.min_gain} in last {self.patience} trials.")
                study.stop()

# function to estimate short and long term models
def forecast_lgbm_whole_sample_LongShortTerm_w_Optuna(
    dat, days, wd, price_s_lags, da_lag, reg_names,
    fuel_lags, n_trials_lgbm=15, days_for_st_model = 31):

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
    
    def rolling_median(series, window):
        output = torch.full_like(series, float('nan'))
        for i in range(window, len(series)):
            output[i] = series[i - window:i].median()
        return output
    
    def pct_change(series):
        result = torch.full_like(series, float('nan'))
        result[1:] = (series[1:] - series[:-1]) / series[:-1] * 100
        return result

    S = dat.shape[1]

    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)
    WD_full = WD.repeat_interleave(S, dim=0)

    reg_names = list(reg_names)
    price_idx = reg_names.index("Price")
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    flat_dat = dat.reshape(-1, dat.shape[2])
    price_series = flat_dat[:, price_idx]
    mat_price_lags = torch.stack([get_lagged(price_series, lag) for lag in price_s_lags], dim=1)

    da_all = []
    for i in da_idx:
        series = flat_dat[:, i].detach().clone()
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    # additional variables
    # "Load_DA" -- pct_change, 
    # "Load_DA" -- lag(168),
    # "Price" -- rolling window, median 2 days
    price_tensor = flat_dat[:, reg_names.index("Price")]
    load_da_tensor = flat_dat[:, reg_names.index("Load_DA")]
    roll2d_median_Price = rolling_median(price_tensor, window=48)
    pct_chg_Load_DA = pct_change(load_da_tensor)
    lag168_Load_DA = get_lagged(load_da_tensor, 168)

    extra_feats_tensor = torch.stack([
        pct_chg_Load_DA,
        lag168_Load_DA,
        roll2d_median_Price
    ], dim=1)

    # join all data
    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, WD_full, mat_fuels, extra_feats_tensor], dim=1)

    # --- feature names ----
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

    feature_names_Xy += ["pct_chg_Load_DA", "lag168_Load_DA", "roll2d_median_Price"]
    # --- ------ ----

    mask = ~torch.isnan(Xy).any(dim=1)
    Xy = Xy[mask]

    n_total = Xy.shape[0]
    last_day_indices = torch.arange(n_total - S, n_total, device=device)

    mean = Xy[:-S].mean(dim=0)
    std = Xy[:-S].std(dim=0)
    std[std == 0] = 1
    Xy_scaled = (Xy - mean) / std

    forecast_x = Xy_scaled[last_day_indices, 1:]
    train_mask = torch.ones(n_total, dtype=torch.bool, device=device)
    train_mask[last_day_indices] = False

    X_train = Xy_scaled[train_mask, 1:].cpu().numpy()
    y_train = Xy_scaled[train_mask, 0].cpu().numpy()
    x_pred = forecast_x.cpu().numpy()

    def make_objective(X_train, y_train, X_val, y_val, feature_names_Xy):
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 256),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 30),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
                "force_row_wise": True,
                "verbosity": -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            x_pred_df = pd.DataFrame(X_val, columns=feature_names_Xy[1:])
            preds = model.predict(x_pred_df)
            return np.mean((preds - y_val) ** 2)
        
        return objective


    # get hyper-parameteres for the long-term model
    # Full size of 24 hours
    val_horizon = S

    # Define validation indices (penultimate day)
    X_val = X_train[-(val_horizon * 2):]
    y_val = y_train[-(val_horizon * 2):]

    # Split for long-term model
    X_long = X_train[:-(val_horizon * 2)]
    y_long = y_train[:-(val_horizon * 2)]

    days_for_st_model = days_for_st_model
    # Split for short-term model
    X_short = X_train[-(val_horizon * days_for_st_model):-(val_horizon * 2)]
    y_short = y_train[-(val_horizon * days_for_st_model):-(val_horizon * 2)]

    # early stopping criteria
    early_stop = StopAfterBestStalls(patience=3, min_gain=0.001)

    # Long-term Optuna
    study_lg = optuna.create_study(direction="minimize")
    study_lg.optimize(make_objective(X_long, y_long, X_val, y_val, feature_names_Xy), 
                    n_trials=n_trials_lgbm, callbacks=[early_stop])
    best_mse_lg = study_lg.best_value

    # Short-term Optuna
    study_st = optuna.create_study(direction="minimize")
    study_st.optimize(make_objective(X_short, y_short, X_val, y_val, feature_names_Xy), 
                    n_trials=n_trials_lgbm, callbacks=[early_stop])
    best_mse_st = study_st.best_value
    
    # Final prediction using tuned long-term model
    final_model_lg = lgb.LGBMRegressor(**study_lg.best_params)
    final_model_lg.fit(np.concatenate([X_long, X_val]), np.concatenate([y_long, y_val]))

    # Final prediction using tuned short-term model
    final_model_st = lgb.LGBMRegressor(**study_st.best_params)
    final_model_st.fit(np.concatenate([X_short, X_val]), np.concatenate([y_short, y_val]))

    x_pred_df = pd.DataFrame(x_pred, columns=feature_names_Xy[1:])

    def weighted_average_predictions(models, losses, x_pred_df):
        inverse_losses = 1 / np.array(losses)
        weights = inverse_losses / inverse_losses.sum()
        preds = np.array([model.predict(x_pred_df) for model in models])
        weighted_preds = np.average(preds, axis=0, weights=weights)
        return weighted_preds
    
    lg_st_models, losses_lg_st = [final_model_lg, final_model_st], [best_mse_lg, best_mse_st]
    y_pred = weighted_average_predictions(lg_st_models, losses_lg_st, x_pred_df)
    y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()

    # feature importance
    for idx,m in enumerate([final_model_lg, final_model_st]):
        importances = m.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": feature_names_Xy[1:],  # skip the raw "Price" if your model did
            "importance": importances
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

        # Print
        print('long model' if idx ==0  else 'short model', '************************')
        print(feature_importance_df)

    return {
        "model": lg_st_models,
        "n_features": X_train.shape[1],
        "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device),
        "losses": losses_lg_st
    }

# ierative comparison; previous, current and combinated models
def forecast_lgbm_whole_sample_optuna_selectBestOptions(
        dat, days, wd, price_s_lags, da_lag, reg_names,
        fuel_lags, ls_models_lgm, n_iteration, n_trials_lgbm=15):

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
    
    def rolling_median(series, window):
        output = torch.full_like(series, float('nan'))
        for i in range(window, len(series)):
            output[i] = series[i - window:i].median()
        return output
    
    def pct_change(series):
        result = torch.full_like(series, float('nan'))
        result[1:] = (series[1:] - series[:-1]) / series[:-1] * 100
        return result

    S = dat.shape[1]
    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)
    WD_full = WD.repeat_interleave(S, dim=0)

    reg_names = list(reg_names)
    price_idx = reg_names.index("Price")
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    flat_dat = dat.reshape(-1, dat.shape[2])
    price_series = flat_dat[:, price_idx]
    mat_price_lags = torch.stack([get_lagged(price_series, lag) for lag in price_s_lags], dim=1)

    da_all = []
    for i in da_idx:
        series = flat_dat[:, i].detach().clone()
        lagged = torch.stack([get_lagged(series, lag) for lag in da_lag], dim=1)
        da_all.append(lagged)
    da_all_var = torch.cat(da_all, dim=1)

    mat_fuel_input = flat_dat[:, fuel_idx]
    mat_fuels = torch.cat([get_lagged_2d(mat_fuel_input, lag) for lag in fuel_lags], dim=1)

    if isinstance(price_series, np.ndarray):
        price_series = torch.tensor(price_series, dtype=torch.float32, device=device)

    # additional variables
    # "Load_DA" -- pct_change, 
    # "Load_DA" -- lag(168),
    # "Price" -- rolling window, median 2 days
    price_tensor = flat_dat[:, reg_names.index("Price")]
    load_da_tensor = flat_dat[:, reg_names.index("Load_DA")]
    roll2d_median_Price = rolling_median(price_tensor, window=48)
    pct_chg_Load_DA = pct_change(load_da_tensor)
    lag168_Load_DA = get_lagged(load_da_tensor, 168)

    extra_feats_tensor = torch.stack([
        pct_chg_Load_DA,
        lag168_Load_DA,
        roll2d_median_Price
    ], dim=1)

    # join all data
    Xy = torch.cat([price_series.unsqueeze(1), mat_price_lags, da_all_var, WD_full, mat_fuels, extra_feats_tensor], dim=1)

    mask = ~torch.isnan(Xy).any(dim=1)
    Xy = Xy[mask]

    # --- feature names ----
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

    feature_names_Xy += ["pct_chg_Load_DA", "lag168_Load_DA", "roll2d_median_Price"]
    # --- ------ ----

    n_total = Xy.shape[0]
    last_day_indices = torch.arange(n_total - S, n_total, device=device)

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

    def objective(trial):
        params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 30),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.05),
        "force_row_wise": True,
        "verbosity": -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train[:-(S*2), :], y_train[:-(S*2)])
        preds = model.predict(X_train[-(S*2):, :])
        y_true = y_train[-(S*2):]
        return np.mean((preds - y_true) ** 2)

    study = optuna.create_study(direction="minimize")
    early_stop = StopAfterBestStalls(patience=3, min_gain=0.001)
    study.optimize(objective, n_trials=n_trials_lgbm, callbacks=[early_stop])
    best_params = study.best_params
    x_pred_df = pd.DataFrame(x_pred, columns=feature_names_Xy[1:])
    lgb_model_current = lgb.LGBMRegressor(**best_params)

    if n_iteration == 0:
        lgb_model_current.fit(X_train, y_train)
        y_pred = lgb_model_current.predict(x_pred_df)
        y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()
        ls_models_lgm.append((lgb_model_current,))
        return {
            "model": lgb_model_current,
            "n_features": X_train.shape[1],
            "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device),
            "forecasts_lgbmOpt": torch.tensor(y_pred, dtype=torch.float32, device=device),
            "study": study,
            "ls_models_lgm": ls_models_lgm
        }

    else:

        def weighted_average_predictions(models, losses, x_pred_df):
            inverse_losses = 1 / np.array(losses)
            weights = inverse_losses / inverse_losses.sum()
            preds = np.array([model.predict(x_pred_df) for model in models])
            weighted_preds = np.average(preds, axis=0, weights=weights)
            return weighted_preds
        
        def weighted_average_predictions_w_pred(preds, losses):
            inverse_losses = 1 / np.array(losses)
            weights = inverse_losses / inverse_losses.sum()
            preds = np.array(preds)
            weighted_preds = np.average(preds, axis=0, weights=weights)
            return weighted_preds
        
        # Full size of 24 hours
        val_horizon = S

        # Define validation indices (penultimate day)
        X_val = X_train[-(val_horizon * 2):]
        y_val = y_train[-(val_horizon * 2):]

        # Split for train before validation
        X_tmp = X_train[:-(val_horizon * 2)]
        y_tmp = y_train[:-(val_horizon * 2)]

        # get predition for validation dataset using  current model
        lgb_model_current.fit(X_tmp, y_tmp)
        preds_current_model = lgb_model_current.predict(X_val)
        mse_current_model = study.best_value

        # get the best previuos model
        select_best_previous_model = ls_models_lgm[n_iteration - 1]
        if not isinstance(select_best_previous_model, tuple):
            select_best_previous_model = (select_best_previous_model,)

        # need to get the weight for the predictions
        ls_preds_best_previous_tmp = [md.predict(X_tmp) for md in select_best_previous_model]
        ls_mse_best_previous_tmp = [np.mean((pred_ - y_tmp) ** 2) for pred_ in ls_preds_best_previous_tmp]
        inverse_losses_ = 1 / np.array(ls_mse_best_previous_tmp)
        weights_ = inverse_losses_ / inverse_losses_.sum()
        
        # estimate prediction of best previous model
        ls_preds_best_previous_model = np.array([md.predict(X_val) for md in select_best_previous_model])
        weighted_preds_best_previous_model = np.average(ls_preds_best_previous_model, axis=0, weights=weights_)
        mse_best_previous_model = np.mean((weighted_preds_best_previous_model - y_val) ** 2)

        # get combinated model using current and previous best models
        avg_pred_current_best = (preds_current_model + weighted_preds_best_previous_model) / 2
        mse_avg_models = np.mean((avg_pred_current_best - y_val) ** 2)

        # get index of best option
        id_min_mse = np.argmin([mse_current_model, mse_best_previous_model, mse_avg_models])
        print(f"Iteration {n_iteration}: MSEs → current: {mse_current_model:.4f}, previous: {mse_best_previous_model:.4f}, avg: {mse_avg_models:.4f}")

        if id_min_mse == 0:
            best_model_iteration = (lgb_model_current,)
            ls_preds_final_model = [md.predict(x_pred_df) for md in best_model_iteration]
            y_pred = np.mean(ls_preds_final_model, axis=0)
            y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()
            print('✅ Current model was best')
        elif id_min_mse == 1:
            best_model_iteration = select_best_previous_model
            ls_preds_best_previous_model_onedayahead = np.array([md.predict(x_pred_df) for md in best_model_iteration])
            y_pred = np.average(ls_preds_best_previous_model_onedayahead, axis=0, weights=weights_)
            y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()
            print('✅ Previous model was best')
        else:
            best_model_iteration = (lgb_model_current,) + select_best_previous_model
            ls_preds_best_previous_model_onedayahead = np.array([md.predict(x_pred_df) for md in select_best_previous_model])
            y_pred_prev = np.average(ls_preds_best_previous_model_onedayahead, axis=0, weights=weights_)
            y_pred_cur = lgb_model_current.predict(x_pred_df) 
            y_pred = weighted_average_predictions_w_pred(preds = [y_pred_cur, y_pred_prev ], 
                                                         losses = [mse_current_model, mse_best_previous_model])
            y_pred = (y_pred * std[0].cpu().item()) + mean[0].cpu().item()
            print('✅ Combination was best')

        # save best option and update the list of models
        ls_models_lgm.append(best_model_iteration)

        return {
            "model": lgb_model_current,
            "n_features": X_train.shape[1],
            "forecasts": torch.tensor(y_pred, dtype=torch.float32, device=device),
            "study": study,
            "ls_models_lgm": ls_models_lgm
        }

