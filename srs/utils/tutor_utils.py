# Copyright 2020 © Michal Narajewski, Florian Ziel


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

# Tutor's code
def DST_trafo(X, Xtime, tz="CET"):
    """Converts a time series DataFrame to a DST-adjusted array

    The function takes a DataFrame of D*S rows and N columns and returns
    an array of shape (D,S,N) where D is the number of days, S the number
    of observations per day and N the number of variables. The function deals
    with the DST problem by averaging the additional hour in October and
    interpolating the missing hour in March.

    Parameters
    ----------
    X : DataFrame
        The time series DataFrame of shape (D*S,N) to be DST-adjusted.
    Xtime : datetime Series
        The series of length D*S containing UTC dates corresponding to the
        DataFrame X.
    tz : str
        The timezone to which the data needs to be adjusted to. The current
        implementation was not tested with other timezones than CET.

    Returns
    -------
    ndarray
        an ndarray of DST-adjusted variables of shape (D,S,N).
    """
    Xinit = X.values
    if len(Xinit.shape) == 1:
        Xinit = np.reshape(Xinit, (len(Xinit), 1))
    atime_init = Xtime.dt.tz_convert('UTC').astype('int64') # <- added this part
    #atime_init = pd.to_numeric(Xtime)
    freq = atime_init.diff().value_counts().idxmax()
    S = int(24*60*60 * 10**9 / freq)
    atime = pd.DataFrame(
        np.arange(start=atime_init.iloc[0], stop=atime_init.iloc[-1]+freq,
                  step=freq))
    idmatch = atime.reset_index().set_index(0).loc[atime_init, "index"].values
    X = np.empty((len(atime), Xinit.shape[1]))
    X[:] = np.nan
    X[idmatch] = Xinit

    new_time = Xtime.dt.tz_convert(tz).reset_index(drop=True)
    DLf = new_time.dt.strftime("%Y-%m-%d").unique()
    days = pd.Series(pd.to_datetime(DLf))

    # EUROPE
    DST_SPRING = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        30) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST_FALL = pd.to_numeric(days.dt.strftime("%m%w")).eq(
        100) & pd.to_numeric(days.dt.strftime("%d")).ge(25)
    DST = ~(DST_SPRING | DST_FALL)

    time_start = new_time.iloc[range(
        S+int(S/24))].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    time_end = new_time.iloc[range(-S-int(S/24), 0)
                             ].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    Dlen = len(DLf)
    Shift = 2  # for CET

    X_dim = X.shape[1]

    Xout = np.empty((Dlen, S, X_dim))
    Xout[:] = np.nan

    k = 0
    # first entry:
    i_d = 0
    idx = time_start[time_start.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, S-1-idx[::-1], ] = X[range(k, len(idx)+k), ]
    elif DST_SPRING[i_d]:
        tmp = S-1-idx[::-1]
        # MARCH
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] == Shift * S/24-1:
                Xout[i_d, range(int(S-S/24 - len(idx) + i_S+1),
                                int(S-S/24 - len(idx) + i_S+1 + S/24)),
                     ] = X[[k+i_S, ]] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if tmp[i_S] > Shift * S/24-1:
                Xout[i_d, int(S-S/24 - len(idx) + i_S+S/24), ] = X[k+i_S, ]
    else:
        tmp = S-idx[::-1]
        # OCTOBER
        for i_S in range(len(idx)):
            if tmp[i_S] <= Shift * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S), ] = X[k+i_S, ]
            if tmp[i_S] in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, int(S+S/24 - len(idx)+i_S), ] = 0.5 * \
                    (X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if tmp[i_S] > (Shift+2) * S/24-1:
                Xout[i_d, int(S+S/24 - len(idx) + i_S-S/24), ] = X[k+i_S, ]
    k += len(idx)
    for i_d in range(1, len(DLf)-1):
        if DST[i_d]:
            idx = S
            Xout[i_d, range(idx), ] = X[range(k, k+idx), ]
        elif DST_SPRING[i_d]:
            idx = int(S-S/24)
            # MARCH
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S == Shift * S/24-1:
                    Xout[i_d, range(int(i_S+1),
                                    int(i_S + 1 + S/24)),
                         ] = X[[k+i_S, ]] + np.transpose(
                        np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                            S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
                if i_S > Shift * S/24-1:
                    Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
        else:
            idx = int(S+S/24)
            # October
            for i_S in range(idx):
                if i_S <= Shift * S/24-1:
                    Xout[i_d, i_S, ] = X[k+i_S, ]
                if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                    Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
                if i_S > (Shift+2) * S/24-1:
                    Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
        k += idx
    # last
    i_d = len(DLf)-1
    idx = time_end[time_end.str.contains(DLf[i_d])].index
    if DST[i_d]:
        Xout[i_d, range(len(idx)), ] = X[range(k, k+len(idx)), ]
    elif DST_SPRING[i_d]:
        # MARCH
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S == Shift * S/24-1:
                Xout[i_d, range(int(i_S+1),
                                int(i_S + 1 + S/24)), ] = X[[k+i_S, ]
                                                            ] + np.transpose(
                    np.atleast_2d(np.arange(1, int(S/24)+1)/(len(range(int(
                        S/24)))+1))).dot(X[[k+i_S+1, ]]-X[[k+i_S, ]])
            if i_S > Shift * S/24-1:
                Xout[i_d, int(i_S + S/24), ] = X[k+i_S, ]
    else:
        # OCTOBER
        for i_S in range(len(idx)):
            if i_S <= Shift * S/24-1:
                Xout[i_d, i_S, ] = X[k+i_S, ]
            if i_S in (Shift*S/24-1 + np.arange(1, int(S/24)+1)):
                Xout[i_d, i_S, ] = 0.5*(X[k+i_S, ] + X[int(k+i_S+S/24), ])
            if i_S > (Shift+2) * S/24-1:
                Xout[i_d, int(i_S-S/24), ] = X[k+i_S, ]
    return Xout

def prepare_dataset_tensor_modified(
    csv_path: Union[str, Path],
    tz: str = "CET",
    seed: int = 42,
    test_days: int = 2 * 365,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, pd.Series, torch.Tensor]:
    """End‑to‑end replication of the notebook preprocessing.

    Parameters
    ----------
    csv_path : str | Path
        Path to `data_no1.csv` (or similar) generated via `merge_data`.
    tz : str, default "CET"
        Local timezone abbreviation for the bidding zone.
    seed : int, default 42
        Seed for all RNGs to ensure reproducibility (matches tutor settings).
    test_days : int, default 730 (≈2 years)
        Number of trailing days reserved for evaluation.
    dtype : torch.dtype, default ``torch.float64``
        Precision of the returned tensors (matches tutor’s use).

    Returns
    -------
    data_tensor : torch.Tensor
        Full `(days, 24, vars)` tensor on the selected device.
    train_tensor : torch.Tensor
        Tensor excluding the last `test_days` days.
    train_dates  : pandas.Series
        Local‑time dates corresponding to `train_tensor` rows.
    price_train  : torch.Tensor
        Slice `[..., 0]` (price) of `train_tensor` for convenience.
    """

    # Deterministic environment & device
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(Path(csv_path))
    time_utc = pd.to_datetime(df["time_utc"], utc=True, format="%Y-%m-%d %H:%M:%S")

    time_lt = time_utc.dt.tz_convert(tz)
    data_array = DST_trafo(X=df.iloc[:, 1:], Xtime=time_utc, tz=tz)
    data_tensor = torch.tensor(data_array, dtype=dtype, device=device)
    price_tensor = data_tensor[..., 0]

    if test_days >= data_tensor.shape[0]:
        raise ValueError("test_days must be smaller than the dataset length")
    
    train_tensor = data_tensor # data_tensor[:-test_days]

    # Build local‑time date index parallel to tensor rows
    local_dates = pd.Series(time_lt.dt.date.unique())
    train_dates = local_dates # local_dates[:-test_days]

    return data_tensor, train_tensor, train_dates, price_tensor

# Bekzod. Helper function for DST_trafo fnc
def prepare_dataset_tensor(
    csv_path: Union[str, Path],
    tz: str = "CET",
    seed: int = 42,
    test_days: int = 2 * 365,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, pd.Series, torch.Tensor]:
    """End‑to‑end replication of the notebook preprocessing.

    Parameters
    ----------
    csv_path : str | Path
        Path to `data_no1.csv` (or similar) generated via `merge_data`.
    tz : str, default "CET"
        Local timezone abbreviation for the bidding zone.
    seed : int, default 42
        Seed for all RNGs to ensure reproducibility (matches tutor settings).
    test_days : int, default 730 (≈2 years)
        Number of trailing days reserved for evaluation.
    dtype : torch.dtype, default ``torch.float64``
        Precision of the returned tensors (matches tutor’s use).

    Returns
    -------
    data_tensor : torch.Tensor
        Full `(days, 24, vars)` tensor on the selected device.
    train_tensor : torch.Tensor
        Tensor excluding the last `test_days` days.
    train_dates  : pandas.Series
        Local‑time dates corresponding to `train_tensor` rows.
    price_train  : torch.Tensor
        Slice `[..., 0]` (price) of `train_tensor` for convenience.
    """

    # Deterministic environment & device
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(Path(csv_path))
    time_utc = pd.to_datetime(df["time_utc"], utc=True, format="%Y-%m-%d %H:%M:%S")

    time_lt = time_utc.dt.tz_convert(tz)
    data_array = DST_trafo(X=df.iloc[:, 1:], Xtime=time_utc, tz=tz)
    data_tensor = torch.tensor(data_array, dtype=dtype, device=device)
    price_tensor = data_tensor[..., 0]

    if test_days >= data_tensor.shape[0]:
        raise ValueError("test_days must be smaller than the dataset length")
    
    train_tensor = data_tensor[:-test_days]

    # Build local‑time date index parallel to tensor rows
    local_dates = pd.Series(time_lt.dt.date.unique())
    train_dates = local_dates[:-test_days]

    return data_tensor, train_tensor, train_dates, price_tensor

# Tutor's code
def get_pbas(Bindex, period=365.24, dK=365.24/6, order=4):
    """Estimates periodic B-splines to model the annual periodicity

    Parameters
    ----------
    Bindex : array_like of int
        The array of day numbers for which to estimate the B-splines.
    period : float
        The period of B-splines. By default set to 365.24.
    dK : float
        The equidistance distance used to calculate the knots.
    order : int
        The order of the B-splines. 3 indicates quadratic splines, 4 cubic etc.

    Returns
    -------
    ndarray
        an ndarray of estimated B-splines.
    """
    # ord=4 --> cubic splines
    # dK = equidistance distance
    # support will be 1:n
    n = len(Bindex)
    stp = dK
    x = np.arange(1, period)  # must be sorted!
    lb = x[0]
    ub = x[-1]
    knots = np.arange(lb, ub+stp, step=stp)
    degree = order-1
    Aknots = np.concatenate(
        (knots[0] - knots[-1] + knots[-1-degree:-1], knots,
         knots[-1] + knots[1:degree+1] - knots[0]))

    from bspline import Bspline
    bspl = Bspline(Aknots, degree)
    basisInterior = bspl.collmat(x)
    basisInteriorLeft = basisInterior[:, :degree]
    basisInteriorRight = basisInterior[:, -degree:]
    basis = np.column_stack(
        (basisInterior[:, degree:-degree],
         basisInteriorLeft+basisInteriorRight))
    ret = basis[np.array(Bindex % basis.shape[0], dtype="int"), :]
    return ret

# Tutor's code
def dm_test(error_a, error_b, hmax=1, power=1):
    # as dm_test with alternative == "less"
    loss_a = (np.abs(error_a)**power).sum(1)**(1/power)
    loss_b = (np.abs(error_b)**power).sum(1)**(1/power)
    delta = loss_a - loss_b
    # estimation of the variance
    delta_var = np.var(delta) / delta.shape[0]
    statistic = delta.mean() / np.sqrt(delta_var)
    delta_length = delta.shape[0]
    k = ((delta_length + 1 - 2 * hmax + (hmax / delta_length)
         * (hmax - 1)) / delta_length)**(1 / 2)
    statistic = statistic * k
    p_value = t.cdf(statistic, df=delta_length-1)

    return {"stat": statistic, "p_val": p_value}

# Tutor's code
def get_cpacf(y, k=1):
    S = y.shape[1]
    n = y.shape[0]
    cpacf = np.full((S, S), np.nan)
    for s in range(S):
        for l in range(S):
            y_s = y[k:n, s]
            y_l_lagged = y[:(n-k), l]
            cpacf[s, l] = np.corrcoef(y_s, y_l_lagged)[0, 1]
    return cpacf

# Tutor's code
def pcor(y, x, z):
    XREG = np.column_stack((np.ones(z.shape[0]), z))
    model_y = LinearRegression(fit_intercept=False).fit(X=XREG, y=y)
    model_x = LinearRegression(fit_intercept=False).fit(X=XREG, y=x)
    cor = np.corrcoef(y - model_y.predict(XREG),
                      x - model_x.predict(XREG))[0, 1]
    return cor

# Tutor's code
def hill(data, start=14, end=None, abline_y=None, ci=0.95, ax=None):
    """Hill estimator translation from R package evir::hill

    Plot the Hill estimate of the tail index of heavy-tailed data, or of an 
    associated quantile estimate.

    Parameters
    ----------
    data : array_like
        data vector
    start : int
        lowest number of order statistics at which to plot a point
    end : int, optional
        highest number of order statistics at which to plot a point
    abline_y : float, optional
        value to be plotted as horizontal straight line
    ci : float
        probability for asymptotic confidence band
    ax : Axes, optional
        the Axes in which to plot the estimator
    """
    ordered = np.sort(data)[::-1]
    ordered = ordered[ordered > 0]
    n = len(ordered)
    k = np.arange(n)+1
    loggs = np.log(ordered)
    avesumlog = np.cumsum(loggs)/k
    xihat = np.hstack([np.nan, (avesumlog-loggs)[1:]])
    alphahat = 1/xihat
    y = alphahat
    ses = y/np.sqrt(k)
    if end is None:
        end = n-1
    x = np.arange(np.min([end, len(data)-1]), start, -1)
    y = y[x]
    qq = norm.ppf(1 - (1-ci)/2)
    u = y + ses[x] * qq
    l = y - ses[x] * qq
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, color='black', linewidth=1)
    ax.plot(x, u, color='red', linestyle='--', linewidth=1)
    ax.plot(x, l, color='red', linestyle='--', linewidth=1)
    if abline_y is not None:
        ax.axhline(abline_y, color='C0', linewidth=1)
    ax.set_ylabel('alpha (CI, p = '+str(ci)+")")
    ax.set_xlabel("Order Statistics")
    
# Tutor's code
def forecast_expert_ext(
    dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
     # Number of hours in a day 
    S = dat.shape[1]  # number of hours

    # Initialize forecast tensor with NaNs for each hour
    forecast = torch.full((S,), float('nan'), device=device)

    # Convert weekday dates to numeric values (1 = Monday, ..., 7 = Sunday)
    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    # Create weekday dummy variables for specified weekdays in `wd`
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)

    # Names of day-ahead forecast variables
    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    
    # Names of fuel variables
    fuel_names = ["Coal", "NGas", "Oil", "EUA"]

    # Get column indices for fuels and DA variables
    reg_names = list(reg_names)
    fuel_idx = torch.tensor([reg_names.index(name) for name in fuel_names], device=device)
    price_idx = reg_names.index("Price")
    da_idx = torch.tensor([reg_names.index(name) for name in da_forecast_names], device=device)

    # Helper function to create 1D lagged tensor
    def get_lagged(Z, lag):
        if lag == 0:
           return Z
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])

    # Helper function to create 2D lagged tensor (for multivariate lags)
    def get_lagged_2d(Z, lag):
        if lag == 0:
           return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=Z.device), Z[:-lag]], dim=0)

    # Create lagged fuel variables for all specified lags and concatenate them
    mat_fuels = torch.cat(
        [get_lagged_2d(dat[:, 0, fuel_idx], lag=l) for l in fuel_lags], dim=1
    )

    # Lagged price from the last hour of the previous day
    price_last = get_lagged(dat[:, S - 1, price_idx], lag=1)

    # Container for coefficients
    num_features = len(wd) + len(price_s_lags) + len(fuel_names)*len(fuel_lags) + len(da_forecast_names)*len(da_lag) + 1
    coefs = torch.full((S, num_features), float('nan'), device=device)
     
     # Loop over each hour of the day to fit a separate regression model 
    for s in range(S):
         # Actual price (target variable) at hour s
        acty = dat[:, s, price_idx]

         # Lagged values of the current hour's price
        mat_price_lags = torch.stack([get_lagged(acty, lag) for lag in price_s_lags], dim=1)

        # Day-ahead forecast values at hour s
        mat_da_forecasts = dat[:, s, da_idx]
        
         # Create lags for each day-ahead forecast variable
        da_lagged_list = [
            torch.stack([get_lagged(mat_da_forecasts[:, i], lag) for lag in da_lag], dim=1)
            for i in range(len(da_forecast_names))
        ]
         # Combine all lagged day-ahead forecasts into one matrix
        da_all_var = torch.cat(da_lagged_list, dim=1)

        # Build the design matrix for regression
        if s == S - 1:
            # For last hour, exclude "Price last" predictor
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels], dim=1
                )
        else:
            # For all other hours, include "Price last"
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels, price_last.unsqueeze(1)], dim=1
                )

        # Filter out rows with missing data
        nan_mask = ~torch.any(torch.isnan(regmat), dim=1)
        regmat0 = regmat[nan_mask]



         # Standardize the data using mean and std of training part
        Xy = regmat0
        mean = Xy[:-1].mean(dim=0)
        std = Xy[:-1].std(dim=0)
        std[std == 0] = 1 # Prevent division by zero
        Xy_scaled = (Xy - mean) / std
          
        # Training data
        X = Xy_scaled[:-1, 1:].cpu().numpy()
        y = Xy_scaled[:-1, 0].cpu().numpy()
        x_pred = Xy_scaled[-1, 1:].cpu().numpy()
        
        # Fit linear regression model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        # Convert coefficients to tensor and clean NaNs
        coef = torch.tensor(model.coef_, dtype=torch.float32, device=device)
        coef[coef != coef] = 0  # Replace NaNs with 0

         # Compute the forecast (re-scale to original units)
        forecast[s] = (coef @ torch.tensor(x_pred, dtype=torch.float32, device=device)) * std[0] + mean[0]

        # Save coefficients
        if s == S - 1:
            coefs[s] = torch.cat([coef, torch.tensor([0.0], device=device)])
        else:
            coefs[s, :coef.numel()] = coef

    # Build coefficient dataframe
    regressor_names = (
        [f"Price lag {lag}" for lag in price_s_lags] +
        [f"{name}_lag_{lag}_s{s}" for name in da_forecast_names for lag in da_lag] +
        [day_abbr[i - 1] for i in wd] +
        [f"{fuel} lag {lag}" for lag in fuel_lags for fuel in fuel_names] +
        ["Price last lag 1"]
    )
    # Convert coefficients to pandas DataFrame for inspection
    coefs_df = pd.DataFrame(coefs.cpu().numpy(), columns=regressor_names)
    
    # Return forecast and coefficients
    return {"forecasts": forecast, "coefficients": coefs_df}

# Tutorial 1:

# Modified by Bekzod
def forecast_expert_ext_modifed(
    dat: torch.Tensor,             # (days, 24, vars)   full history window incl. 'today'
    days: pd.Series,               # pd.DatetimeIndex or Series – same length as dat[:,0,0]
    reg_names: list[str],          # column names  (Price must be index 0 afterwards)
    wd: list[int] = [1, 6, 7],     # weekday dummies
    price_s_lags: list[int] = [1, 2, 7],
    da_lag: list[int] = [0],
    fuel_lags: list[int] = [2],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor | pd.DataFrame]:
    """
    Vectorised re-implementation of the tutor’s ‘expert_ext’ day-ahead model.
    Returns the 24-h forecast plus the coefficient table for inspection.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dat = dat.to(device)

    S = dat.shape[1]                       # hours in day  (24)
    forecast = torch.full((S,), float("nan"), device=device)

    # -- basic index bookkeeping --------------------------------------
    weekdays_num = torch.tensor(days.dt.weekday.values + 1, device=device)
    WD = torch.stack([(weekdays_num == x).float() for x in wd], dim=1)

    da_forecast_names = ["Load_DA", "Solar_DA", "WindOn_DA", "WindOff_DA"]
    fuel_names        = ["Coal", "NGas", "Oil", "EUA"]

    reg_names = list(reg_names)
    price_idx = reg_names.index("Price")
    da_idx    = torch.tensor([reg_names.index(n) for n in da_forecast_names], device=device)
    fuel_idx  = torch.tensor([reg_names.index(n) for n in fuel_names],        device=device)

    # helpers ----------------------------------------------------------
    def lag_1d(x: torch.Tensor, k: int):
        return x if k == 0 else torch.cat([torch.full((k,), float("nan"), device=x.device), x[:-k]])
    def lag_2d(x: torch.Tensor, k: int):
        return x if k == 0 else torch.cat([torch.full((k, x.shape[1]), float("nan"), device=x.device), x[:-k]], dim=0)

    mat_fuels  = torch.cat([lag_2d(dat[:, 0, fuel_idx], k) for k in fuel_lags], dim=1)
    price_last = lag_1d(dat[:, S-1, price_idx], 1)

    # container for coefficients
    n_feat = len(price_s_lags) + len(da_forecast_names)*len(da_lag) + len(wd) + len(fuel_names)*len(fuel_lags) + 1
    coefs  = torch.full((S, n_feat), float("nan"), device=device)

    # hour-by-hour local regressions ----------------------------------
    for s in range(S):
        y = dat[:, s, price_idx]
        price_lags  = torch.stack([lag_1d(y, k) for k in price_s_lags], dim=1)
        da_vals     = dat[:, s, da_idx]                          # (days, 4)
        da_lagged   = torch.cat([torch.stack([lag_1d(da_vals[:,i], k) for k in da_lag], dim=1)
                                 for i in range(len(da_forecast_names))], dim=1)

        X_block = [price_lags, da_lagged, WD, mat_fuels]
        if s != S-1:
            X_block.append(price_last.unsqueeze(1))
        X = torch.cat(X_block, dim=1)
        regmat = torch.cat([y.unsqueeze(1), X], dim=1)

        ok = ~torch.any(torch.isnan(regmat), dim=1)
        regmat_ok = regmat[ok]

        mu     = regmat_ok[:-1].mean(0)
        sigma  = regmat_ok[:-1].std(0)
        sigma[sigma == 0] = 1          
        Z = (regmat_ok - mu) / sigma

        X_train, y_train = Z[:-1, 1:].cpu().numpy(), Z[:-1, 0].cpu().numpy()
        x_pred           = Z[-1, 1:].cpu().numpy()

        beta = LinearRegression(fit_intercept=False).fit(X_train, y_train).coef_
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        beta[beta != beta] = 0.0                                        

        forecast[s] = (beta @ torch.tensor(x_pred, dtype=torch.float32, device=device)) * sigma[0] + mu[0]
        coefs[s, :beta.numel()] = beta

    regressor_names = (
        [f"Price lag {k}" for k in price_s_lags] +
        [f"{n}_lag_{lag}" for n in da_forecast_names for lag in da_lag] +
        [calendar.day_abbr[i-1] for i in wd] +
        [f"{f} lag {lag}"  for lag in fuel_lags for f in fuel_names] +
        ["Price last lag 1"]
    )
    coef_df = pd.DataFrame(coefs.cpu().numpy(), columns=regressor_names)
    return {"forecasts": forecast, "coefficients": coef_df}

# Bekzod. Helper function for forecast_expert_ext_modified
def forecasting_study(
    data_t: torch.Tensor,          
    dates_s: pd.Series,            
    reg_names: list[str],
    length_eval: int   = 2*365,    
    history_days: int  = 730,      
    wd: list[int]      = [1, 6, 7],
    price_s_lags: list[int] = [1, 2, 7],
    da_lag: list[int]  = [0],
    fuel_lags: list[int] = [2],
    save_path: str | Path | None = "OLS/forecasting_study.pt",
    seed: int = 42,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Re-creates the tutor’s walk-forward evaluation loop and (optionally) saves
    the `(length_eval, 24, 2)` tensor with *true* and *expert_ext* forecasts.
    """
    # deterministic GPU / RNG setup  — same as earlier snippet
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.enabled       = False

    data_t = data_t.to(device)
    price_s = data_t[..., 0]         # target slice
    S = data_t.shape[1]
    begin_eval = data_t.shape[0] - length_eval          # first eval day index

    forecasts = torch.full((length_eval, S, 2), float("nan"),
                           dtype=torch.float64, device=device)

    for n in range(length_eval):
        forecasts[n, :, 0] = price_s[begin_eval + n]    # “true” prices

        day_slice = slice(begin_eval - history_days + n,  begin_eval + n + 1)
        dat_hist  = data_t[day_slice]                   # (D+1, 24, vars)
        days_hist = dates_s.iloc[day_slice]

        forecasts[n, :, 1] = forecast_expert_ext_modifed(
            dat        = dat_hist,
            days       = days_hist,
            reg_names  = reg_names,
            wd         = wd,
            price_s_lags = price_s_lags,
            da_lag     = da_lag,
            fuel_lags  = fuel_lags,
            device     = device,
        )["forecasts"]

        if (n+1) % 10 == 0 or n+1 == length_eval:
            pct = 100 * (n+1) / length_eval
            print(f"\r-> {pct:5.1f}%  done", end="", flush=True)

    print()  # newline after progress bar
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(forecasts.cpu(), save_path)
    return forecasts

# Bekzod. Visualization helper functions from tutor's code.
def plot_hour_comparison(
    forecasts: torch.Tensor,          
    hour: int,
    test_dates: pd.Series,            
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """
    Line-plot true vs expert forecast for one hour across the test window.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    true_vals     = forecasts[:, hour, 0].cpu().numpy()
    expert_vals   = forecasts[:, hour, 1].cpu().numpy()

    ax.plot(test_dates, true_vals,  label="True")
    ax.plot(test_dates, expert_vals, label="Expert forecast", alpha=0.7, lw=2)
    ax.set(
        title = title or f"True vs Expert ⟶ hour {hour}",
        xlabel= "Date",
        ylabel= "Price",
    )
    ax.legend(); ax.grid(True); plt.tight_layout()
    return ax

# Bekzod. Visualization helper functions from tutor's code.
def plot_daily_profile(
    forecasts: torch.Tensor,          
    obs_index: int = -1,              
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """
    Plot the 24-hour profile for a single observation index.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    hours = range(24)
    ax.plot(hours, forecasts[obs_index, :, 0].cpu().numpy(),
            label="True", lw=2)
    ax.plot(hours, forecasts[obs_index, :, 1].cpu().numpy(),
            label="Expert forecast", ls="--")

    ax.set(
        xlabel="Hour",
        ylabel="Price",
        title=title or f"Daily profile – obs {obs_index}",
    )
    ax.legend(); ax.grid(True); plt.tight_layout()
    return ax

# Tutorial 2: MULTI-WINDOW EXPERTS  +  EXPONENTIALLY WEIGHTED AVERAGING (EWA)

# Bekzod. 1) Build experts for several history windows                  
def build_multiwindow_experts(
    data_t: torch.Tensor,       # (Days, 24, Vars)
    dates_s: pd.Series,         # length Days  (timezone-aware)
    reg_names: List[str],
    window_sizes: Dict[int, int] = {1:186, 2:365, 3:730},
    wd: List[int] = [1,6,7],
    price_s_lags: List[int] = [1,2,7],
    start_offset: int = 0,      # 0 → begin at end of data_t
    forecast_days: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns a tensor  (forecast_days × 24 × (1+|window_sizes|)):
    channel 0 = true prices, channels 1…k = expert forecasts
    """

    device = device or (torch.device("cuda") if torch.cuda.is_available() else "cpu")
    data_t = data_t.to(device)
    price_s = data_t[...,0]

    # evaluation or test horizon
    if forecast_days is None:
        forecast_days = max(window_sizes.values())      # safe default
    begin = data_t.shape[0] - forecast_days - start_offset

    S = data_t.shape[1]
    n_models = 1 + len(window_sizes)        # true + experts
    forecasts = torch.full((forecast_days, S, n_models),
                           float("nan"), dtype=torch.float64, device=device)

    # main loop
    for n in range(forecast_days):
        # channel 0 : true values
        forecasts[n,:,0] = price_s[begin+n]

        for idx,key in enumerate(window_sizes, start=1):
            win = window_sizes[key]
            slice_hist = slice(begin - win + n, begin + n + 1)
            dat_hist   = data_t[slice_hist]
            days_hist  = dates_s.iloc[slice_hist]

            forecasts[n,:,idx] = forecast_expert_ext_modifed(
                dat        = dat_hist,
                days       = days_hist,
                reg_names  = reg_names,
                wd         = wd,
                price_s_lags = price_s_lags,
                device     = device,
            )["forecasts"]

        if (n+1) % 10 == 0 or n+1 == forecast_days:
            pct = 100*(n+1)/forecast_days
            print(f"\rbuilding experts … {pct:5.1f}% done", end="", flush=True)
    print()
    return forecasts

#  Bekzod. EWA – device-safe, self-contained (no global N_s needed)
def run_ewa_torch(
    K: int,                          # number of experts
    expert_preds: torch.Tensor,      # shape (T, K)
    actuals:      torch.Tensor,      # shape (T,)
    eta: float,
    w0: torch.Tensor | None = None,
    device: torch.device | None = None,
):
    """
    Exponentially-Weighted Average (square-loss) for K experts over T steps.

    Returns
    -------
    w_hist   : (T+1, K)  weight trajectory  (row 0 = initial weights)
    agg_pred : (T,)      aggregated prediction each step
    agg_loss : (T,)      squared loss of aggregator each step
    """
    # -------- device / dtype handling ---------------------------------
    device = device or (expert_preds.device if torch.is_tensor(expert_preds)
                        else torch.device("cpu"))
    expert_preds = expert_preds.to(device)
    actuals      = actuals.to(device)

    T = expert_preds.shape[0]                    # number of time steps

    # -------- initial weights -----------------------------------------
    if w0 is None:
        w0 = torch.ones(K, device=device) / K
    else:
        w0 = w0.to(device)

    # -------- storage -------------------------------------------------
    w_hist   = torch.empty((T + 1, K), device=device)
    w_hist[0] = w0
    agg_pred = torch.empty(T, device=device)
    agg_loss = torch.empty(T, device=device)

    # -------- main loop ----------------------------------------------
    for t in range(T):
        # 1) aggregated forecast
        agg_pred[t] = (w_hist[t] * expert_preds[t]).sum()

        # 2) individual and agg losses (squared error)
        losses   = (expert_preds[t] - actuals[t]) ** 2
        agg_loss[t] = (agg_pred[t] - actuals[t]) ** 2

        # 3) EWA weight update
        w_next = w_hist[t] * torch.exp(-eta * losses)
        denom  = w_next.sum()
        w_hist[t + 1] = w_next / denom if denom > 1e-15 else (1.0 / K)

    return w_hist, agg_pred, agg_loss

# Bekzod. 2) Grid-search η per hour on an evaluation tensor              
def tune_ewa_eta(
    forecasts_eval: torch.Tensor,     # (N_s, 24, 1+k)
    eta_grid: List[float] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns tensor  (24,)  of best η for each hour.
    """
    if eta_grid is None:
        eta_grid = [2**i for i in range(-10,2)]   # 2^-10 … 2^1
    device = device or forecasts_eval.device
    N_s, S, _ = forecasts_eval.shape
    k = forecasts_eval.shape[2]-1

    from math import inf
    best_eta = torch.empty(S, dtype=torch.float32, device=device)

    # helper re-use from below
    for s in range(S):
        expert = forecasts_eval[:,s,1:]   
        true   = forecasts_eval[:,s,0]    
        min_loss = inf
        for eta in eta_grid:
            _,_,losses = run_ewa_torch(k, expert, true, eta, device=device)
            tot = losses.sum().item()
            if tot < min_loss:
                min_loss, best = tot, eta
        best_eta[s] = best
    return best_eta

# Bekzod. 3) Apply hour-specific EWA to a *test* tensor                    
def ewa_aggregate_forecasts(
    forecasts_test: torch.Tensor,   # (N_s, 24, 1+k)
    eta_per_hour: torch.Tensor,     # (24,)
    burn_fraction: float = 1/9,     # burn last ~quarter of eval as in tutor
    device: torch.device | None = None,
):
    """
    Returns:
      agg_preds    – (N_s, 24) aggregated forecast
      agg_sq_loss  – (N_s, 24) squared error of aggregator
      burn_period  – int, number of initial days dropped
    """
    device = device or forecasts_test.device
    N_s, S, k_plus = forecasts_test.shape
    k = k_plus - 1
    burn_period = int(N_s - N_s*burn_fraction)     # same as tutor

    agg_preds   = torch.zeros((N_s,S), device=device)
    agg_sq_loss = torch.zeros((N_s,S), device=device)

    for s in range(S):
        expert  = forecasts_test[:,s,1:]
        true    = forecasts_test[:,s,0]
        _, preds, sq = run_ewa_torch(k, expert, true, eta_per_hour[s], device=device)
        agg_preds[:,s]   = preds
        agg_sq_loss[:,s] = sq

    return agg_preds, agg_sq_loss, burn_period

# 4) Bekzod. Error summary (RMSE / MAE)  + optional hourly RMSE curve      
def compute_error_table(
    forecasts_all: torch.Tensor, 
    model_names: List[str],
    plot_hourly: bool = True,
):
    """
    Builds a pandas table (RMSE & MAE) and – optionally – a per-hour RMSE plot.
    """
    errors = forecasts_all[...,1:] - forecasts_all[...,:1]
    err = errors.cpu().numpy()
    rmse = np.sqrt(np.nanmean(err**2, axis=(0,1)))
    mae  = np.nanmean(np.abs(err),    axis=(0,1))

    df = pd.DataFrame([rmse, mae], index=["RMSE","MAE"], columns=model_names[1:])
    if plot_hourly:
        rmse_hour = np.sqrt(np.nanmean(err**2, axis=0))
        
        print(rmse_hour.shape)              # should be  (24, 4)
        print(np.isnan(rmse_hour).sum())    # how many NaNs in the 96 cells?
        
        # Bekzod. Which hours are still all-NaN?
        bad_hours = [h for h in range(24) if np.isnan(rmse_hour[h]).all()]
        print("Hours with no valid RMSE:", bad_hours)
        
        # Bekzod
        hours = range(24)
        for i, name in enumerate(model_names[1:]):
            curve = rmse_hour[:, i]
            if np.isnan(curve).all():          # hour has no valid cells
                continue                       # skip drawing that curve
            plt.plot(hours, curve, label=name)

        plt.xlabel("Hour"); plt.ylabel("RMSE")
        plt.title("Hourly RMSE – test period")
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout(); plt.show()
    return df

# Tutorial 3: Hyperparameter tuning with Optuna

# Bekzod. Helper: build 1-window expert forecasts for a horizon of N days 
def _build_window_forecasts(
    data_t: torch.Tensor,          # (Days, 24, Vars)
    dates_s: pd.Series,            # len Days
    reg_names: List[str],
    window: int,                   # history length D
    start_idx: int,                # first day index to forecast
    horizon: int,                  # N_s
    wd: List[int],
    price_s_lags: List[int],
    da_lag: List[int],
    fuel_lags: List[int],
    device: torch.device,
) -> torch.Tensor:                 # (horizon, 24, 2)  true + expert

    S = data_t.shape[1]
    out = torch.full((horizon, S, 2), float("nan"),
                     dtype=torch.float64, device=device)
    price = data_t[..., 0]

    for n in range(horizon):
        # slice that forms the training window for this forecast day
        hist_slice = slice(start_idx - window + n, start_idx + n + 1)
        
        # --- TEMP diagnostic at top of _build_window_forecasts loop -----------
        if n == 0:                      # print only for first day
            hist_start = start_idx - window + n
            hist_end   = start_idx + n            # inclusive last index

            print(f"[DEBUG] window D = {window}")
            print(f"[DEBUG] expecting history rows {hist_start} … {hist_end}")
            print(f"[DEBUG] hist_slice actually sent : {hist_slice.start} … {hist_slice.stop-1}")
        # ----------------------------------------------------------------------
        
        # channel 0 = true price
        out[n, :, 0] = price[start_idx + n]
        
        dat_hist   = data_t[hist_slice]
        days_hist  = dates_s.iloc[hist_slice]

        out[n, :, 1] = forecast_expert_ext_modifed(
            dat        = dat_hist,
            days       = days_hist,
            reg_names  = reg_names,
            wd         = wd,
            price_s_lags = price_s_lags,
            da_lag     = da_lag,
            fuel_lags  = fuel_lags,
            device     = device,
        )["forecasts"]
    return out

# Bekzod. 1) Optuna tuning of window length D                               #
def tune_expert_window(
    data_t: torch.Tensor,
    dates_s: pd.Series,
    reg_names: List[str],
    eval_days: int    = 2*365,          # 730-day eval block
    window_bounds: Tuple[int,int] = (186, 730),
    n_trials: int     = 50,
    wd: List[int]     = [1,6,7],
    price_s_lags: List[int] = [1,2,7],
    da_lag: List[int] = [0],
    fuel_lags: List[int] = [2],
    device: torch.device | None = None,
) -> Tuple[int, optuna.study.Study]:
    """
    Returns the best window size D and the Optuna study object.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    data_t = data_t.to(device)

    S  = data_t.shape[1]
    N  = eval_days
    T0 = data_t.shape[0] - eval_days            # first eval day index

    def objective(trial):
        D = trial.suggest_int("D", *window_bounds)
        fc = _build_window_forecasts(
            data_t, dates_s, reg_names,
            window      = D,
            start_idx   = T0,
            horizon     = N,
            wd=wd, price_s_lags=price_s_lags, da_lag=da_lag, fuel_lags=fuel_lags,
            device=device,
        )
        err = (fc[...,1] - fc[...,0]).cpu().numpy()
        rmse = np.sqrt(np.nanmean(err**2))      # scalar
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_D = study.best_params["D"]
    return best_D, study

# Bekzod. 2) Run expert with chosen D on a test horizon                     #
def run_expert_window_test(
    data_t: torch.Tensor,
    dates_s: pd.Series,
    reg_names: List[str],
    window: int,
    test_days: int = 730,
    wd: List[int]     = [1,6,7],
    price_s_lags: List[int] = [1,2,7],
    da_lag: List[int] = [0],
    fuel_lags: List[int] = [2],
    device: torch.device | None = None,
) -> torch.Tensor:                 # (test_days, 24, 2)
    device = device or data_t.device
    start_idx = data_t.shape[0] - test_days
    return _build_window_forecasts(
        data_t, dates_s, reg_names,
        window=window, start_idx=start_idx, horizon=test_days,
        wd=wd, price_s_lags=price_s_lags, da_lag=da_lag,
        fuel_lags=fuel_lags, device=device,
    )

# Build regression matrix (one row = one day)                    
def build_regression_matrix(
    dat_eval: np.ndarray,        # (Days, 24, Vars)
    days_eval: pd.Series,        # len Days   (datetime, local tz)
    reg_names: pd.Index,
    wd: List[int],          
    price_lags: List[int],
    da_lag: List[int],       
    fuel_lags: List[int], 
    da_names: List[str],
    fuel_names: List[str],
) -> Dict[str, any]:
    """
    Re-implements the tutor’s long reg_matrix() in ~50 lines and returns:
      - regmat          : pandas DF   (Days, Features)
      - index_dict      : {series → column-index list}
      - dep_indices     : list of dependent-variable column positions
    """
    S = dat_eval.shape[1]                    # 24

    # helper
    def lag_1d(x, k):
        if k == 0: return x
        return np.concatenate([np.full(k, np.nan), x[:-k]])

    # weekday dummies (WD=[0] => drop WD column)
    if not (len(wd) == 1 and wd[0] == 0):
        weekdays_num = days_eval.dt.weekday + 1 
        WD = np.column_stack([(weekdays_num == d).astype(int) for d in wd])
        wd_cols = [f"WD_{d}" for d in wd]
    else:
        WD = np.empty((dat_eval.shape[0], 0))
        wd_cols = []

    # fuel lags
    fuels = dat_eval[:, 0, reg_names.isin(fuel_names)]
    fuel_block = np.concatenate(
        [np.apply_along_axis(lag_1d, 0, fuels, k) for k in fuel_lags],
        axis=1)
    fuel_cols = [f"{f}_lag_{k}" for k in fuel_lags for f in fuel_names]
    
    # assemble base block    
    base_block = np.column_stack([WD, fuel_block])
    base_cols  = wd_cols + fuel_cols
    base_df    = pd.DataFrame(base_block, columns=base_cols)

    # per-series part
    per_series = []
    for s in range(S):
        price = dat_eval[:, s, reg_names == "Price"].ravel()
        price_lag_block = np.column_stack([lag_1d(price,k) for k in price_lags])
        
        # day-ahead values and their lags
        da_block = []
        da_vals  = dat_eval[:, s, reg_names.isin(da_names)]
        for i in range(len(da_names)):
            da_block.append(np.column_stack([lag_1d(da_vals[:,i], k) for k in da_lag]))
        da_block = np.hstack(da_block)

        block = np.column_stack([price, price_lag_block, da_block])
        cols  = ([f"Price_s{s}"] +
                 [f"Price_lag_{k}_s{s}" for k in price_lags] +
                 [f"{n}_lag_{k}_s{s}"  for n in da_names for k in da_lag])
        per_series.append(pd.DataFrame(block, columns=cols))

    per_df = pd.concat(per_series, axis=1)
    regmat = pd.concat([per_df, base_df], axis=1)

    # build index structures
    columns_s   = per_series[0].shape[1]
    columns_base= base_df.shape[1]

    index_dict   = {s: list(range(s*columns_s, (s+1)*columns_s))
                    + list(range(regmat.shape[1]-columns_base,
                                 regmat.shape[1]))
                    for s in range(S)}
    dep_indices  = [idx[0] for idx in index_dict.values()]   # first col per series

    return {
        "regmat"      : regmat,
        "index_dict"  : index_dict,
        "dep_indices" : dep_indices,
    }

# Build regression matrix (one row = one day)                    
def build_regression_matrix_modified(
    dat_eval: np.ndarray,        # (Days, 24, Vars)
    days_eval: pd.Series,        # len Days   (datetime, local tz)
    reg_names: pd.Index,
    wd: List[int],               # weekday dummies to include, [0] disables
    price_lags: List[int],       # lags of spot price to use as regressors
    da_lag: List[int],           # lags of DA series
    fuel_lags: List[int],        # lags of fuel series
    da_names: List[str],         # column names of DA series
    fuel_names: List[str],       # column names of fuel series
) -> Dict[str, Any]:
    """
    Builds a regression matrix with:
      - demeaned hourly spot prices as targets (P_center_s{h})
      - P_daily_mean as a regressor
      - price lags, DA lags, fuel lags and weekday dummies as regressors
      - returns regmat (Days × Features), index_dict, and dep_indices.
    """
    
    n_days, S, n_vars = dat_eval.shape
    price_idx = reg_names.get_loc("Price")
    df = pd.DataFrame({
        f"Price_s{h}": dat_eval[:, h, price_idx]
        for h in range(S)
    }, index=days_eval)
    
    # daily mean and demean
    df["P_daily_mean"] = df.mean(axis=1)
    demean_cols = []
    # for h in range(S):
    #     center = f"P_center_s{h}"
    #     df[center] = df[f"Price_s{h}"] - df["P_daily_mean"]
    #     demean_cols.append(center)
    # regmat_df = df.drop(columns=[f"Price_s{h}" for h in range(S)])

    # helper for lags
    def lag_1d(x, k):
        if k == 0:
            return x
        return np.concatenate([np.full(k, np.nan), x[:-k]])

    # weekday dummies
    if not (len(wd) == 1 and wd[0] == 0):
        weekdays_num = days_eval.dt.weekday + 1
        WD = np.column_stack([(weekdays_num == d).astype(int) for d in wd])
        wd_cols = [f"WD_{d}" for d in wd]
        wd_df = pd.DataFrame(WD, index=days_eval, columns=wd_cols)
    else:
        wd_df = pd.DataFrame(index=days_eval)

    # fuel lags (using horizon 0 as representative)
    fuels = dat_eval[:, 0, reg_names.isin(fuel_names)]
    fuel_block = np.concatenate(
        [np.apply_along_axis(lag_1d, 0, fuels, k) for k in fuel_lags], axis=1
    )
    fuel_cols = [f"{f}_lag_{k}" for k in fuel_lags for f in fuel_names]
    fuel_df = pd.DataFrame(fuel_block, index=days_eval, columns=fuel_cols)

    # per-horizon regressors: price lags and DA lags
    per_list = []
    for h in range(S):
        spot = dat_eval[:, h, price_idx]
        # daily mean and demean
        center = f"P_center_s{h}"
        df[center] = df[f"Price_s{h}"] - df["P_daily_mean"]
        demean_cols.append(center)
        # price lags
        pl = np.column_stack([lag_1d(spot, k) for k in price_lags])
        pl_cols = [f"Price_lag_{k}_s{h}" for k in price_lags]
        # DA lags
        da_vals = dat_eval[:, h, reg_names.isin(da_names)]
        da_block = np.hstack([
            np.column_stack([lag_1d(da_vals[:, i], k) for k in da_lag])
            for i in range(len(da_names))
        ])
        da_cols = [f"{n}_lag_{k}_s{h}" for n in da_names for k in da_lag]
        df_h = pd.DataFrame(
            np.column_stack([pl, da_block]),
            index=days_eval,
            columns=pl_cols + da_cols
        )
        per_list.append(df_h)
    regmat_df = df.drop(columns=[f"Price_s{h}" for h in range(S)])
    per_df = pd.concat(per_list, axis=1)

    # final regression matrix
    regmat = pd.concat([regmat_df, per_df, wd_df, fuel_df], axis=1)

    # index_dict: map each horizon h to the position of its demeaned target
    index_dict = {
        h: [regmat.columns.get_loc(f"P_center_s{h}")]
        for h in range(S)
    }
    # dep_indices: list of all demeaned target positions
    dep_indices = [idxs[0] for idxs in index_dict.values()]

    return {
        "regmat": regmat,
        "index_dict": index_dict,
        "dep_indices": dep_indices,
    }

# with reordering resolved and data leakage issue
def build_regression_matrix_modified_2(
    dat_eval: np.ndarray,        # (Days, 24, Vars)
    days_eval: pd.Series,        # len Days   (datetime, local tz)
    reg_names: pd.Index,
    wd: List[int],               # weekday dummies to include, [0] disables
    price_lags: List[int],       # lags of spot price to use as regressors
    da_lag: List[int],           # lags of DA series
    fuel_lags: List[int],        # lags of fuel series
    da_names: List[str],         # column names of DA series
    fuel_names: List[str],       # column names of fuel series
) -> Dict[str, Any]:
    """
    Builds a regression matrix with columns ordered by horizon blocks:
      - P_daily_mean first
      - For each h in 0..S-1: [P_center_s{h}, Price_lag_*_s{h}, DA_lag_*_s{h}]
      - Then weekday dummies, then fuel lags
    Returns regmat (Days × Features), index_dict, and dep_indices.
    """
    n_days, S, _ = dat_eval.shape
    # locate variable indices
    price_idx = reg_names.get_loc("Price")

    # compute daily mean of raw hourly prices
    prices = np.stack([dat_eval[:, h, price_idx] for h in range(S)], axis=1)
    P_daily_mean = prices.mean(axis=1)
    # prepare P_daily_mean DF
    df_mean = pd.DataFrame({"P_daily_mean": P_daily_mean}, index=days_eval)

    # helper for lags
    def lag_1d(x, k):
        if k == 0:
            return x
        return np.concatenate([np.full(k, np.nan), x[:-k]])

    # build per-horizon blocks including demean + lags
    per_list = []
    index_dict = {}
    for h in range(S):
        # raw spot for horizon h
        spot_h = dat_eval[:, h, price_idx]
        # demeaned target
        center = spot_h - P_daily_mean
        # price lag features
        pl_block = np.column_stack([lag_1d(spot_h, k) for k in price_lags])
        pl_cols = [f"Price_lag_{k}_s{h}" for k in price_lags]
        # DA lag features
        da_vals = dat_eval[:, h, reg_names.isin(da_names)]
        da_block = np.hstack([
            np.column_stack([lag_1d(da_vals[:, i], k) for k in da_lag])
            for i in range(len(da_names))
        ])
        da_cols = [f"{n}_lag_{k}_s{h}" for n in da_names for k in da_lag]
        # assemble block DF
        cols_h = [f"P_center_s{h}"] + pl_cols + da_cols
        data_h = np.column_stack([center.reshape(-1,1), pl_block, da_block])
        df_h = pd.DataFrame(data_h, index=days_eval, columns=cols_h)
        # record index of target within the full concatenation
        # will be position = len(df_mean.columns) + sum of widths of previous blocks
        index_dict[h] = [None]  # placeholder, will set after concat
        per_list.append(df_h)

    # weekday dummies
    if not (len(wd) == 1 and wd[0] == 0):
        weekdays_num = days_eval.dt.weekday + 1
        WD = np.column_stack([(weekdays_num == d).astype(int) for d in wd])
        wd_cols = [f"WD_{d}" for d in wd]
        wd_df = pd.DataFrame(WD, index=days_eval, columns=wd_cols)
    else:
        wd_df = pd.DataFrame(index=days_eval)

    # fuel lags at horizon 0
    fuels = dat_eval[:, 0, reg_names.isin(fuel_names)]
    fuel_block = np.concatenate(
        [np.apply_along_axis(lag_1d, 0, fuels, k) for k in fuel_lags], axis=1
    )
    fuel_cols = [f"{f}_lag_{k}" for k in fuel_lags for f in fuel_names]
    fuel_df = pd.DataFrame(fuel_block, index=days_eval, columns=fuel_cols)

    # concatenate all pieces in desired order
    all_blocks = [df_mean] + per_list + ([wd_df] if not wd_df.empty else []) + ([fuel_df] if not fuel_df.empty else [])
    regmat = pd.concat(all_blocks, axis=1)

    # now fill in index_dict with actual positions
    col_index = df_mean.shape[1]
    for h, df_h in enumerate(per_list):
        # position of P_center_s{h}
        idx = col_index
        index_dict[h] = [idx]
        # advance by number of columns in this block
        col_index += df_h.shape[1]
    # dep_indices as ordered list
    dep_indices = [index_dict[h][0] for h in range(S)]

    return {
        "regmat": regmat,
        "index_dict": index_dict,
        "dep_indices": dep_indices,
    }

# with both reordering and data leakage resolved
def build_regression_matrix_reordered_dtleak(
    dat_eval: np.ndarray,        # (Days, 24, Vars)
    days_eval: pd.Series,        # len=Days (datetime, local tz)
    reg_names: pd.Index,         # length=Vars
    wd: List[int],               # weekday dummies to include; [0] disables
    price_lags: List[int],       # e.g. [1,2,7]
    da_lag: List[int],           # e.g. [0]
    fuel_lags: List[int],        # e.g. [2]
    da_names: List[str],         # ["Load_DA","Solar_DA",…]
    fuel_names: List[str],       # ["Coal","NGas",…]
    price_names: List[str],      # subset of ['P_center','P_daily_mean']
) -> Dict[str, Any]:
    """
    Builds regression matrix with:
      • optional lag-1 daily mean ('P_daily_mean')
      • optional 24-dim demeaned-target ('P_center_s{h}')
      • per-horizon price-lags & DA-lags, interleaved by horizon
      • weekday dummies (if wd != [0])
      • fuel lags (once, at horizon 0)

    Returns:
      - regmat     : DataFrame (Days × Features)
      - index_dict : {horizon → [col-index of P_center_s{h}]}
      - dep_indices: list of those positions
    """

    # dims
    n_days, S, _ = dat_eval.shape
    price_idx    = reg_names.get_loc("Price")

    # stack raw hourly prices: shape (Days, S)
    prices = np.stack([dat_eval[:, h, price_idx] for h in range(S)], axis=1)

    # today’s mean (for demean target) & lag-1 mean (for regressor)
    today_mean = prices.mean(axis=1)
    def lag_1d(x: np.ndarray, k: int) -> np.ndarray:
        if k == 0: return x
        return np.concatenate([np.full(k, np.nan), x[:-k]])
    mean_lag1 = lag_1d(today_mean, 1)

    # prepare the single-column daily-mean DF if requested
    blocks: List[pd.DataFrame] = []
    if "P_daily_mean" in price_names:
        df_mean = pd.DataFrame(
            {"P_daily_mean": mean_lag1},
            index=days_eval
        )
        blocks.append(df_mean)

    # build per-horizon blocks (interleaving P_center, price-lags, DA-lags)
    per_list: List[pd.DataFrame] = []
    for h in range(S):
        spot_h = prices[:, h]
        cols, data = [], []

        # 1) demeaned target
        if "P_center" in price_names:
            cols.append(f"P_center_s{h}")
            data.append(spot_h - today_mean)

        # 2) price lags
        for k in price_lags:
            cols.append(f"Price_lag_{k}_s{h}")
            data.append(lag_1d(spot_h, k))

        # 3) DA lags
        da_vals = dat_eval[:, h, reg_names.isin(da_names)]
        for i, name in enumerate(da_names):
            for k in da_lag:
                cols.append(f"{name}_lag_{k}_s{h}")
                data.append(lag_1d(da_vals[:, i], k))

        # assemble this horizon’s DF
        per_list.append(pd.DataFrame(
            np.column_stack(data),
            index=days_eval,
            columns=cols
        ))

    # weekday dummies
    if not (len(wd) == 1 and wd[0] == 0):
        weekdays_num = days_eval.dt.weekday + 1
        WD = np.column_stack([(weekdays_num == d).astype(int) for d in wd])
        wd_df = pd.DataFrame(
            WD, index=days_eval, columns=[f"WD_{d}" for d in wd]
        )
    else:
        wd_df = pd.DataFrame(index=days_eval)

    # fuel lags (horizon-0 representative)
    fuel_vals = dat_eval[:, 0, reg_names.isin(fuel_names)]
    fuel_block = np.concatenate([
        lag_1d(fuel_vals[:, i], k).reshape(-1,1)
        for k in fuel_lags for i in range(fuel_vals.shape[1])
    ], axis=1)
    fuel_cols = [f"{n}_lag_{k}" for k in fuel_lags for n in fuel_names]
    fuel_df   = pd.DataFrame(fuel_block, index=days_eval, columns=fuel_cols)

    # concatenate in the exact order we want:
    blocks += per_list
    if not wd_df.empty:    blocks.append(wd_df)
    if not fuel_df.empty:  blocks.append(fuel_df)
    regmat = pd.concat(blocks, axis=1)

    # build index_dict & dep_indices for the P_center_s{h} targets
    index_dict: Dict[int, List[int]] = {}
    for h in range(S):
        idx = regmat.columns.get_loc(f"P_center_s{h}")
        index_dict[h] = [idx]
    dep_indices = [index_dict[h][0] for h in range(S)]

    return {
        "regmat":      regmat,
        "index_dict":  index_dict,
        "dep_indices": dep_indices,
    }


# Prepare tensors for one forecast date                          
def prepare_train_test_tensors(
    regmat_df: pd.DataFrame,
    dep_indices: List[int],
    D: int,
    eval_start_row: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with X_train, y_train, X_test, y_test, mean_x, std_x, mean_y, std_y
    for one evaluation step (the tutor’s “first obs in evaluation set”).
    """
    regmat_df = regmat_df.dropna().reset_index(drop=True)
    regmat = torch.tensor(regmat_df.values, dtype=torch.float32, device=device)
    dep    = regmat[:, dep_indices]

    # zero-out dep in features
    regmat[:, dep_indices] = 0

    mean_x = regmat[eval_start_row - D : eval_start_row].mean(0, keepdim=True)
    std_x  = regmat[eval_start_row - D : eval_start_row].std(0, keepdim=True)
    std_x[std_x == 0] = 1

    X_train = (regmat[eval_start_row - D : eval_start_row] - mean_x) / std_x
    X_test  = ((regmat[eval_start_row] - mean_x) / std_x).unsqueeze(0)

    mean_y = dep[eval_start_row - D : eval_start_row].mean(0, keepdim=True)
    std_y  = dep[eval_start_row - D : eval_start_row].std(0, keepdim=True)
    std_y[std_y == 0] = 1

    y_train = (dep[eval_start_row - D : eval_start_row] - mean_y) / std_y
    y_test  = ((dep[eval_start_row] - mean_y) / std_y).unsqueeze(0)

    return dict(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
    )


