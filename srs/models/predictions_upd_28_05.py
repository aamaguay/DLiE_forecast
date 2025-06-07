 # %% load packages
import locale
from math import pi
import os
import pandas as pd
import numpy as np
from tutorials.my_functions import DST_trafo, forecast_expert_ext, run_forecast_step, forecast_gam, forecast_gam_whole_sample
import polars as pl
import matplotlib.pyplot as plt
import optuna
import requests
import torch
import random
from sqlalchemy import create_engine,inspect
from pathlib import Path
import urllib.parse
from pygam import LinearGAM, s
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

#%%
import importlib
import tutorials.my_functions as my_functions

# Reload the module to get the latest version of functions
importlib.reload(my_functions)

# Access the functions from the reloaded module
DST_trafo = my_functions.DST_trafo
forecast_expert_ext = my_functions.forecast_expert_ext
run_forecast_step = my_functions.run_forecast_step
forecast_gam = my_functions.forecast_gam
forecast_gam_whole_sample = my_functions.forecast_gam_whole_sample

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
    "expert_ext",
    "linar_gam"
]

# Total number of models
n_models = len(model_names)

# Initialize a 3D tensor to hold forecasts:
forecasts = torch.full((N_s, S, n_models), float('nan'), dtype=torch.float64, device=device)

#%%
# run the process using a loop
# Start timing
init_time = datetime.now()
# Loop over each forecasting step
for n in range(N_s)[:2]:

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

    # Generate forecasts using one model for the whole dataset
    forecasts[n, :, 2] = forecast_gam_whole_sample(
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

# End timing
end_time = datetime.now()

# Compute duration
duration = end_time - init_time
duration_minutes = duration.total_seconds() / 60
print(f"Training duration: {duration_minutes:.2f} minutes")


#%%
# run procrss using ThreadPoolExecutor
# run process
# Start timing
init_time = datetime.now()

# Allocate results container
results = [None] * N_s

# Create thread pool
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            run_forecast_step,
            n,
            price_S,
            data_array,
            begin_eval,
            D,
            dates_S,
            wd,
            price_s_lags,
            da_lag,
            data.columns[1:],  # reg_names
            data.columns[1:]   # data_columns
        )
        for n in range(N_s)
    ]

    for future in as_completed(futures):
        try:
            n, true_price, expert, gam = future.result()
            forecasts[n, :, 0] = true_price
            forecasts[n, :, 1] = torch.tensor(expert, dtype=forecasts.dtype, device=forecasts.device)
            forecasts[n, :, 2] = torch.tensor(gam, dtype=forecasts.dtype, device=forecasts.device)
        except Exception as e:
            print(f"Thread crashed: {e}")

# End timing
end_time = datetime.now()
duration_minutes = (end_time - init_time).total_seconds() / 60
print(f"\nParallel training duration (threaded): {duration_minutes:.2f} minutes")

#%%
# estimate rmse for all models, validation dataset
true_values = forecasts[:, :, 0] 

# Add a new axis to true_values to allow broadcasting
true_expanded = true_values.unsqueeze(-1) 

# Repeat along last dim
FFT = true_expanded.repeat(1, 1, forecasts.shape[2]) 
squared_errors = (FFT - forecasts) ** 2  


# Average squared error over all days and hours (dim=0 and dim=1)
mse_per_model = squared_errors.mean(dim=(0, 1))

# Take square root to get RMSE per model
rmse_per_model = torch.sqrt(mse_per_model) 


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
# chart for last obs. of the validation data
last_obs = 200
plt.figure(figsize=(10, 4))
plt.plot(forecasts[:, :, 0].flatten().cpu().numpy()[-last_obs:], label="True", linewidth=2)
plt.plot(forecasts[:, :, 1].flatten().cpu().numpy()[-last_obs:], label="Expert Forecast", linestyle="--")
plt.plot(forecasts[:, :, 2].flatten().cpu().numpy()[-last_obs:], label="GAM", linestyle=":")
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# chart for validation data
plt.figure(figsize=(10, 4))
plt.plot(forecasts[:, :, 0].flatten().cpu().numpy(), label="True", linewidth=2)
plt.plot(forecasts[:, :, 1].flatten().cpu().numpy(), label="Expert Forecast", linestyle="--")
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

# Initialize a 3D tensor to hold forecasts:
forecasts_test = torch.full((N_s, S, n_models), float('nan'), dtype=torch.float64, device=device)


#%%#####################################################
############################Forecats test##############
#######################################################
# using loop approach
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


#%% 
# run process using ThreadPoolExecutor approach
# Start timing
init_time = datetime.now()

# Allocate results container
results_test = [None] * N_s

# Create thread pool
with ThreadPoolExecutor() as executor:
    futures_test = [
        executor.submit(
            run_forecast_step,
            n,
            price_S,
            data_array,
            begin_test,
            D,
            dates_S,
            wd,
            price_s_lags,
            da_lag,
            data.columns[1:],  # reg_names
            data.columns[1:]   # data_columns
        )
        for n in range(N_s)
    ]

    for future in as_completed(futures_test):
        try:
            n, true_price, expert, gam = future.result()
            forecasts_test[n, :, 0] = true_price
            forecasts_test[n, :, 1] = torch.tensor(expert, dtype=forecasts.dtype, device=forecasts.device)
            forecasts_test[n, :, 2] = torch.tensor(gam, dtype=forecasts.dtype, device=forecasts.device)
        except Exception as e:
            print(f"Thread crashed: {e}")

# End timing
end_time = datetime.now()
duration_minutes = (end_time - init_time).total_seconds() / 60
print(f"\nParallel training duration (threaded): {duration_minutes:.2f} minutes")

#%%
# estimate rmse for all models, test dataset
true_values_test = forecasts_test[:, :, 0] 

# Add a new axis to true_values to allow broadcasting
true_expanded_test = true_values_test.unsqueeze(-1) 

# Repeat along last dim
FFT_test = true_expanded_test.repeat(1, 1, forecasts_test.shape[2]) 
squared_errors_test = (FFT_test - forecasts_test) ** 2  

# Get the mask of valid rows (no NaNs across any hour or model)
valid_rows_mask = ~torch.isnan(squared_errors_test).any(dim=(1, 2)) 
squared_errors_test = squared_errors_test[valid_rows_mask]

# Average squared error over all days and hours (dim=0 and dim=1)
mse_per_model_test = squared_errors_test.mean(dim=(0, 1))

# Take square root to get RMSE per model
rmse_per_model_test = torch.sqrt(mse_per_model_test) 

# %%
# chart for test data, last obs
last_obs_ = 300
plt.figure(figsize=(10, 4))
plt.plot(forecasts_test[:, :, 0].flatten().cpu().numpy()[-last_obs_:], label="True", linewidth=2)
plt.plot(forecasts_test[:, :, 1].flatten().cpu().numpy()[-last_obs_:], label="Expert Forecast", linestyle="--")
plt.plot(forecasts_test[:, :, 2].flatten().cpu().numpy()[-last_obs_:], label="GAM", linestyle=":")
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()