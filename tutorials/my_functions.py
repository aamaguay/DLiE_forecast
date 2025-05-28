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
import torch
import random
from pygam import LinearGAM, s
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    atime_init = pd.to_numeric(Xtime)
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


def pcor(y, x, z):
    XREG = np.column_stack((np.ones(z.shape[0]), z))
    model_y = LinearRegression(fit_intercept=False).fit(X=XREG, y=y)
    model_x = LinearRegression(fit_intercept=False).fit(X=XREG, y=x)
    cor = np.corrcoef(y - model_y.predict(XREG),
                      x - model_x.predict(XREG))[0, 1]
    return cor


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
    
    
def forecast_expert_ext(
    dat, days, wd, price_s_lags, da_lag, reg_names, fuel_lags):
     # Number of hours in a day 
    S = dat.shape[1]  # number of hours

    # print(f"shape of initial data: {dat.shape}........ ")
    # print(f"missing values of initial data: {np.isnan(dat).sum(axis=(0, 1))}........ ")

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

    # print('matfuels........................')
    # print(f"fuel name ids ..{fuel_idx}")
    # print(dat[:, 0, fuel_idx])

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
        # print(mat_price_lags.shape)
        # print(mat_price_lags)

        # Day-ahead forecast values at hour s
        mat_da_forecasts = dat[:, s, da_idx]
        
         # Create lags for each day-ahead forecast variable
        da_lagged_list = [
            torch.stack([get_lagged(mat_da_forecasts[:, i], lag) for lag in da_lag], dim=1)
            for i in range(len(da_forecast_names))
        ]
         # Combine all lagged day-ahead forecasts into one matrix
        da_all_var = torch.cat(da_lagged_list, dim=1)
        # print(f"....{s}.....")
        # print(da_all_var.shape)
        # print(da_all_var)

        # Build the design matrix for regression
        if s == S - 1:
            # For last hour, exclude "Price last" predictor
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels], dim=1
                )
            # print('option1')
        else:
            # For all other hours, include "Price last"
            regmat = torch.cat(
                [acty.unsqueeze(1), mat_price_lags, da_all_var, WD, mat_fuels, price_last.unsqueeze(1)], dim=1
               )
            # print('option2')
        # print(f"....{s}..hereeeee result reg...")
        # print(regmat.shape) 
        # print(regmat)
        # print("Missing values per column before we remove:", torch.isnan(regmat).sum(dim=0))

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
        # print(f"********{s},....{Xy.shape}")
        # print(Xy)
        # print(X.shape, len(y), Xy.shape)
        # print(X)
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
        return torch.cat([torch.full((lag,), float('nan'), device=Z.device), Z[:-lag]])

    def get_lagged_2d(Z, lag):
        if lag == 0:
            return Z
        return torch.cat([torch.full((lag, Z.shape[1]), float('nan'), device=Z.device), Z[:-lag]], dim=0)

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



def run_forecast_step(n, price_S, data_array, begin_eval, D, dates_S, wd, price_s_lags, da_lag, reg_names, data_columns):
    print(f"START NS: {n}")
    
    # Save true price
    true_price = price_S[begin_eval + n]
    
    # Get days and data slice
    days = pd.to_datetime(dates_S[(begin_eval - D + n):(begin_eval + 1 + n)])
    dat_slice = data_array[(begin_eval - D + n):(begin_eval + 1 + n)]

    # GAM forecast
    gam_forecast = forecast_gam_whole_sample(
        dat=dat_slice,
        days=days,
        wd=wd,
        price_s_lags=price_s_lags,
        da_lag=da_lag,
        reg_names=data_columns,
        fuel_lags=[2]
    )["forecasts"]

    # Expert model forecast
    expert_forecast = forecast_expert_ext(
        dat=dat_slice,
        days=days,
        wd=wd,
        price_s_lags=price_s_lags,
        da_lag=da_lag,
        reg_names=data_columns,
        fuel_lags=[2]
    )["forecasts"]

    print(f"END NS: {n}")
    return n, true_price, expert_forecast, gam_forecast
