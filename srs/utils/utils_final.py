

# load packages
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
from typing import List, Dict, Any

#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

# ------------------------------------------------------------------ #
# basic utility
def lag_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return x
    return np.concatenate((np.full(k, np.nan), x[:-k]))

# ------------------------------------------------------------------ #
# 1) weekday dummies
def make_weekday_dummies(days: pd.Series, wd: List[int]) -> Tuple[np.ndarray, List[str]]:
    if len(wd) == 1 and wd[0] == 0:
        return np.empty((len(days), 0)), []
    wnum = days.dt.weekday + 1
    mat  = np.column_stack([(wnum == d).astype(int) for d in wd])
    cols = [f"WD_{d}" for d in wd]
    return mat, cols

# ------------------------------------------------------------------ #
# 2) fuel features (day level)
def make_fuel_features(dat: np.ndarray, reg: pd.Index,
                       names: List[str], lags: List[int]
                      ) -> Tuple[np.ndarray, List[str]]:
    base = dat[:, 0, reg.isin(names)]
    block = np.concatenate([np.apply_along_axis(lag_1d, 0, base, k) for k in lags], axis=1)
    cols  = [f"{n}_lag_{k}" for k in lags for n in names]
    return block, cols

# ------------------------------------------------------------------ #
# 3) seasonal numbers + trigs (day level)
def make_seasonal_features(days: pd.Series,
                           num_names: List[str],
                           trig_names: List[str]
                          ) -> Tuple[np.ndarray, List[str]]:
    blocks, cols = [], []
    hr      = days.dt.hour.to_numpy()
    hr_week = (days.dt.weekday * 24 + hr).to_numpy()
    hr_year = ((days.dt.dayofyear - 1) * 24 + hr).to_numpy()

    mapper = {
        "D_t":      (hr,        24),
        "W_t":      (hr_week,  168),
        "A_t":      (hr_year, 8760),
        "W_t_sin":  (np.sin(2*np.pi*hr_week/168), None),
        "W_t_cos":  (np.cos(2*np.pi*hr_week/168), None),
        "A_t_sin":  (np.sin(2*np.pi*hr_year/8760), None),
        "A_t_cos":  (np.cos(2*np.pi*hr_year/8760), None),
    }

    for key in num_names + trig_names:
        if key not in mapper:
            continue
        arr, _ = mapper[key]
        blocks.append(arr[:, None])
        cols.append(key)
    if blocks:
        return np.hstack(blocks), cols
    return np.empty((len(days), 0)), []

# ------------------------------------------------------------------ #
# 4) volatility (day level)
def make_volatility_features(dat: np.ndarray, reg: pd.Index,
                             names: List[str]
                            ) -> Tuple[np.ndarray, List[str]]:
    if not names:
        return np.empty((dat.shape[0], 0)), []
    p_ix   = np.where(reg == "Price")[0][0]
    flat_p = dat.reshape(-1, dat.shape[2])[:, p_ix]
    blocks, cols = [], []
    if "vol_24h_lag1" in names:
        v = pd.Series(flat_p).rolling(24).std().shift(1).to_numpy().reshape(dat.shape[0], 24)[:,0]
        blocks.append(v[:, None]);  cols.append("vol_24h_lag1")
    if "volpct_24h_lag1" in names:
        pct = pd.Series(flat_p).pct_change()
        v   = pct.rolling(24).std().shift(1).to_numpy().reshape(dat.shape[0], 24)[:,0]
        blocks.append(v[:, None]);  cols.append("volpct_24h_lag1")
    return np.hstack(blocks), cols

# ------------------------------------------------------------------ #
# 5) target creation
def make_target(prices_flat: np.ndarray, mode: str = "hour") -> np.ndarray:
    if mode == "hour":
        return np.concatenate(([np.nan], np.diff(prices_flat)))
    else:  # "day"
        return prices_flat - lag_1d(prices_flat, 24)

# ------------------------------------------------------------------ #
# 6) price-diff lag features (per-hour)
def make_diff_lag_cube(diff_flat: np.ndarray, n_days: int, lags: List[int],
                       unit: str = "h") -> np.ndarray:
    step = 24 if unit == "d" else 1
    cube = [lag_1d(diff_flat, k*step).reshape(n_days, 24) for k in lags]
    return np.stack(cube, axis=-1)          # (Days, 24, |lags|)

# ------------------------------------------------------------------ #
# 7) misc hour-level extras for one hour slice
def make_hour_extras(h: int, n_days: int,
                     price_level: np.ndarray,
                     load_da: np.ndarray
                    ) -> Tuple[np.ndarray, List[str]]:
    # sin/cos of hour
    sin_h = np.full(n_days, np.sin(2*np.pi*h/24))
    cos_h = np.full(n_days, np.cos(2*np.pi*h/24))
    # rush-hour dummy
    rush  = np.full(n_days, int(7<=h<=9 or 17<=h<=19))
    # pct-chg & lag168 of Load_DA
    pct_ld = np.concatenate(([np.nan], np.diff(load_da)/load_da[:-1]*100))
    lag168_ld = lag_1d(load_da, 168)
    # 2nd diff of price & lag1
    sec_diff = np.concatenate(([np.nan, np.nan], np.diff(np.diff(price_level))))
    lag1_sec = lag_1d(sec_diff, 1)

    block = np.column_stack([sin_h, cos_h, rush, pct_ld, lag168_ld, lag1_sec])
    cols  = ["sin_hour", "cos_hour", "hour_7to9_17to19_dummy",
             "pct_chg_Load_DA", "lag168_Load_DA", "lag1_price_2nd_diff"]
    return block, cols

# ------------------------------------------------------------------ #
def build_regression_matrix(
    dat_eval: np.ndarray,
    days_eval: pd.Series,
    reg_names: pd.Index,
    *,
    wd: List[int],
    price_lags: List[int] | None = None,
    da_lags: List[int],
    fuel_lags: List[int],
    da_names: List[str],
    fuel_names: List[str],
    seasonal_names: List[str],
    seasonal_sin_cos_names: List[str],
    vol_names: List[str],
    diff_price_lags: List[int],
    target_mode: str = "hour",     # "hour" | "day"
    lag_unit: str   = "h",         # "h"    | "d"
) -> Dict[str, Any]:

    S, n_days = dat_eval.shape[1], dat_eval.shape[0]
    price_ix  = np.where(reg_names == "Price")[0][0]

    # ========= day-level blocks =========
    WD,  WD_cols  = make_weekday_dummies(days_eval, wd)
    FUEL, F_cols  = make_fuel_features(dat_eval, reg_names, fuel_names, fuel_lags)
    SEAS, S_cols  = make_seasonal_features(days_eval, seasonal_names, seasonal_sin_cos_names)
    VOL,  V_cols  = make_volatility_features(dat_eval, reg_names, vol_names)

    base_block = np.column_stack([WD, FUEL, SEAS, VOL])
    base_cols  = WD_cols + F_cols + S_cols + V_cols
    base_df    = pd.DataFrame(base_block, columns=base_cols)

    # ========= target & diff lags =========
    prices_flat = dat_eval.reshape(-1, dat_eval.shape[2])[:, price_ix]
    diff_flat   = make_target(prices_flat, mode=target_mode)
    diff_cube   = make_diff_lag_cube(diff_flat, n_days, diff_price_lags, unit=lag_unit)

    if price_lags is None: price_lags = []
    
    # ========= per-hour blocks =========
    per_frames = []
    for h in range(S):
        # target
        y_h = diff_cube[:, h, 0] if diff_cube.ndim == 3 else diff_flat.reshape(n_days, S)[:, h]

        # level price lags
        level = dat_eval[:, h, price_ix]
        if price_lags: lvl_lags = np.column_stack([lag_1d(level, k) for k in price_lags])
        else: lvl_lags = np.empty((n_days, 0))

        # diff lags
        diff_lags_h = diff_cube[:, h, :] if diff_cube.ndim == 3 else np.empty((n_days, 0))

        # DA forecasts
        da_vals = dat_eval[:, h, reg_names.isin(da_names)]
        da_blk = np.hstack([np.column_stack([lag_1d(da_vals[:, i], k) for k in da_lags])
                            for i in range(len(da_names))])

        # hour extras
        load_da = dat_eval[:, h, reg_names == "Load_DA"].ravel()
        extras, extras_cols = make_hour_extras(h, n_days, level, load_da)

        block = np.column_stack([y_h, lvl_lags, diff_lags_h, da_blk, extras])
        cols  = (
            [f"del_price_s{h}"] +
            ([f"Price_lag_{k}_s{h}" for k in price_lags] if price_lags else []) +
            [f"del_price_lag{lag}{lag_unit}_s{h}" for lag in diff_price_lags] +
            [f"{n}_lag_{k}_s{h}" for n in da_names for k in da_lags] +
            [c + f"_s{h}" for c in extras_cols]
        )
        per_frames.append(pd.DataFrame(block, columns=cols))

    per_df = pd.concat(per_frames, axis=1)
    regmat = pd.concat([per_df, base_df], axis=1)

    cols_per_h = per_frames[0].shape[1]
    cols_base  = base_df.shape[1]
    idx_dict   = {h: list(range(h*cols_per_h, (h+1)*cols_per_h)) +
                       list(range(regmat.shape[1]-cols_base, regmat.shape[1]))
                  for h in range(S)}
    dep_idx    = [idx[0] for idx in idx_dict.values()]

    return {"regmat": regmat, "index_dict": idx_dict, "dep_indices": dep_idx}
