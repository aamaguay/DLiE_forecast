# %% load packages
import locale
import os
import pandas as pd
import numpy as np
from tutorials.my_functions import DST_trafo, forecast_expert_ext
import polars as pl
import matplotlib.pyplot as plt
import optuna
import requests
import torch
import random
from sqlalchemy import create_engine,inspect
from pathlib import Path
import urllib.parse

#%%
import importlib
import tutorials.my_functions as my_functions

# Reload the module to get the latest version of functions
importlib.reload(my_functions)

# Access the functions from the reloaded module
DST_trafo = my_functions.DST_trafo
forecast_expert_ext = my_functions.forecast_expert_ext

#%%
#set the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set CuBLAS deterministic behavior to enforce deterministic behavior for CuBLAS operations when using optuna
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# for cuda
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# %%
# import dataframe
data = pd.read_csv('./tutorials/data_no1.csv')


# Convert the 'time_utc' column to timezone-aware datetime objects (in UTC) using the specified format
time_utc = pd.to_datetime(data["time_utc"],
                          utc=True, format="%Y-%m-%d %H:%M:%S")

# convert time_utc to local time
local_time_zone = "CET"  # local time zone abbrv
time_lt = time_utc.dt.tz_convert(local_time_zone)
time_lt

S = 24

#%% Save the start and end-time
start_end_time_S = time_lt.iloc[[0, -1]].dt.tz_localize(None).dt.tz_localize("UTC")

# creating 'fake' local time
start_end_time_S_num = pd.to_numeric(start_end_time_S)
time_S_numeric = np.arange(
    start=start_end_time_S_num.iloc[0],
    stop=start_end_time_S_num.iloc[1] + 24 * 60 * 60 * 10**9 / S,
    step=24 * 60 * 60 * 10**9 / S,
)

#  'fake' local time
time_S = pd.Series(pd.to_datetime(time_S_numeric, utc=True))
dates_S = pd.Series(time_S.dt.date.unique())


#%% import DST_trafo function and use it on data
data_array = DST_trafo(X=data.iloc[:, 1:], Xtime=time_utc, tz=local_time_zone)
type(data_array)
print(f"original data has: {data.shape} (including time_utc), and then after DST we have: {data_array.shape}, .... and it is: {data_array.shape[0]*data_array.shape[1] } obs... ")

 
#%% Change data_array to tensor
data_array= torch.tensor(data_array, dtype=torch.float64, device=device)

# save the prices as dataframe
price_S = data_array[..., 0]

# Keep the last 2 years for test
N = 2 * 365
dat_eval = data_array[:-N,:,:]
days_eval = pd.to_datetime(dates_S)[:-N]

Price_eval = price_S[:-N]

#%%###########################################################################
####################function for forecats#################################
##########################################################################
# Define the validation period length
length_eval = 2 * 365

# The first obdervation in the evaluation period
begin_eval = dat_eval.shape[0] - length_eval

N_s = length_eval

D = 730

# Specify the weekday dummies: Mon, Sat, and Sun 
wd = [1, 6, 7]

#Specify the lags: lag 1 , lag 2 and lag 7
price_s_lags = [1, 2, 7]

#Specify lags of DA: predictions of tomorrow 
da_lag = [0]

#%%####################################################################
######################## Forecasting Study  for the Evaluation data####
#######################################################################

# List of model names 
model_names = [
    "true",
    "expert_ext"
]

# Total number of models
n_models = len(model_names)

# Initialize a 3D tensor to hold forecasts:
forecasts = torch.full((N_s, S, n_models), float('nan'), dtype=torch.float64, device=device)

# Loop over each forecasting step
for n in range(N_s):

    print(f"********************* START NS ... {n} ****************************************************")
    # Save the actual ("true") prices for evaluation
    forecasts[n, :, 0] = price_S[begin_eval+n]

      # Select the date range for the current window (D days of history + current day)
    days = pd.to_datetime(dates_S[(begin_eval - D + n) : (begin_eval + 1+ n)])

    # Generate forecasts using expert_ext model and save them
    forecasts[n, :, 1] = forecast_expert_ext(
        dat = data_array[(begin_eval - D + n) : (begin_eval + 1 + n)], 
        days = days,
        wd = wd,
        price_s_lags = price_s_lags,
        da_lag = da_lag,
        reg_names = data.columns[1:],
        fuel_lags = [2])["forecasts"]

     # Progress tracker (as percentage)
    progress = torch.tensor((n + 1) / N * 100, dtype=torch.float64)
    print(f"\r-> {progress.item():.2f}% done", end="")

    print(f"\n********************* END NS.. {n} ****************************************************")


# %%#####################################################################
#######################Comparison Plots##################################
#######################################################################

#select the hour, chart for a specific hour
hour = 14

# Select the actual and forecasted prices for the specific hour
true_values = forecasts[:, hour, 0].cpu().numpy()
forecast_values = forecasts[:, hour, 1].cpu().numpy()


#Specify the dates of the test data
dates_x = days_eval[-N:]
# Line plot comparison
plt.figure(figsize=(10, 5))
plt.plot(dates_x, true_values, label="True")
plt.plot(dates_x, forecast_values, label="Forecast (Expert)", alpha=0.7, linewidth=2)
plt.title(f"Forecast vs True Values at hour {hour} Across Test Data")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Select observation index, chart for one day 
obs = 729
plt.figure(figsize=(10, 4))
plt.plot(forecasts[obs, :, 0].cpu().numpy(), label="True", linewidth=2)
plt.plot(forecasts[obs, :, 1].cpu().numpy(), label="Expert Forecast", linestyle="--")
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# chart for validation data
plt.figure(figsize=(10, 4))
plt.plot(forecasts[:, :, 0].flatten().cpu().numpy()[-800:], label="True", linewidth=2)
plt.plot(forecasts[:, :, 1].flatten().cpu().numpy()[-800:], label="Expert Forecast", linestyle="--")
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%#######################################################
############## Test Data ###############
###########################################################

# Use all data including the test data
days_test = pd.to_datetime(dates_S)
dat_test = data_array

#%%#######################################################
##############Define test data###############
##########################################################

#  Define the test period length
length_test = (2 * 365 )

# The first obdervation in the evaluation period
begin_test = days_test.shape[0] - length_test

N_s = length_test


#%%#####################################################
############################Forecats test##############
#######################################################

# Initialize a 3D tensor to hold forecasts:
forecasts_test = torch.full((N_s, S, n_models), float('nan'), dtype=torch.float64, device=device)

# Loop over each forecasting step
for n in range(N_s):

    print(f"********************* START NS ... {n} ****************************************************")

    
    # Save the actual ("true") prices for comparison later
    forecasts_test[n, :, 0] = price_S[begin_test+n]

    # Select the date range for the current window 
    days = pd.to_datetime(dates_S[(begin_test - D + n) : (begin_test + 1+ n)])

    # Generate forecasts using expert_ext model and save them
    forecasts_test[n, :, 1] = forecast_expert_ext(
        dat=data_array[(begin_test - D + n) : (begin_test + 1 + n)], 
        days=days, 
        reg_names=data.columns[1:],
        wd=wd, 
        da_lag = da_lag,
        price_s_lags=price_s_lags,
        fuel_lags = [2]
    )["forecasts"]

     # Progress tracker (as percentage)
    progress = torch.tensor((n + 1) / N * 100, dtype=torch.float64)
    print(f"\r-> {progress.item():.2f}% done", end="")
    print(f"\n********************* END NS.. {n} ****************************************************")



# %%
# chart for test data
plt.figure(figsize=(10, 4))
plt.plot(forecasts_test[:, :, 0].flatten().cpu().numpy()[-800:], label="True", linewidth=2)
plt.plot(forecasts_test[:, :, 1].flatten().cpu().numpy()[-800:], label="Expert Forecast", linestyle="--")
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
