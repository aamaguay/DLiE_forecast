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

#%%######################################################################
#                   Data from ENTSOE
#########################################################################

# Have a look at https://transparency.entsoe.eu/content/static_content/Static%20content/knowledge%20base/knowledge%20base.html
# for more information about ENTSOE

# Define MySQL connection parameters
db_user = "student"                             # Username for the MySQL database
db_password = "#q6a21I&OA5k"                     # Password for the MySQL user (empty here)
host = "132.252.60.112"                          # Hostname or IP address of the MySQL server
port = 3306                                      # Port number MySQL is listening on
dbname = "ENTSOE"                                # Name of the database you want to connect to

#  %%Create MySQL engine
engine = create_engine(
    f"mysql://{urllib.parse.quote_plus(db_user)}:{urllib.parse.quote_plus(db_password)}@{host}:{port}/{dbname}"
)

#%%Check the available tables in engine
inspector = inspect(engine)
tables = inspector.get_table_names()
print(tables)

#%% SQL query to select all rows from the 'spec' table
spec_query = "SELECT * FROM spec"
# Execute the query and load results into a DataFrame
spec = pd.read_sql_query(spec_query, engine)

#%% Get an overview
print(spec.head())

#%%Market divisions: BZN=bidding zones,...
spec.MapTypeCode.unique()

#%%Country code: DE_LU = Germany and Luxembourg, FR = France...
# here i just need, All Norwegian zones (NO1,...,NO5), 'NO1', 'NO2', 'NO3', 'NO4', 'NO5
spec.MapCode.unique()
spec.MapCode[spec.MapCode.fillna('').str.startswith('NO')].unique()

#%% Retrieve the resulution of the time series:PT60M= hourly data..
spec.ResolutionCode.unique()

# %%
# create function to download the data
def download_data(targets, engine, yr, yr_end):
    # Obtain the forecasts (and actuals) from the vals table
    # Build a SQL query string that:
    #  - Selects all columns from the vals table
    #  - Filters to only those TimeSeriesID values we found in `targets`
    #  - Restricts the data to year 2014 and onward
    values_query = f"""
    SELECT *
    FROM vals
    WHERE TimeSeriesID IN ({", ".join(map(str, targets['TimeSeriesID']))})
    AND YEAR(`DateTime`) >= '{yr}' AND YEAR(`DateTime`) < '{yr_end}'
    """
    # Execute the SQL and load the results into a DataFrame
    values = pd.read_sql_query(values_query, engine)

    # Merge the time-series values with their corresponding spec 
    # metadata on TimeSeriesID so each row has both the data point 
    # and its descriptive attributes
    data = pd.merge(values, targets, on="TimeSeriesID")

    return data

# function to sum observations per hour
def custom_sum(series):
    return series.sum() if series.notna().any() else np.nan

# Group by and aggregate using sum (respecting your custom_sum logic)
# If your `custom_sum` only sums non-NaNs or returns NaN if all NaN:
def custom_sum_polars(series: pl.Series):
    return None if series.null_count() == len(series) else series.sum()

def custom_mean_polars(series: pl.Series):
    return None if series.null_count() == len(series) else series.mean()


# %% ************** GENERATION ******************************
# Filter the spec DataFrame down to the specific time series you want:
targets_generation = spec[
    (spec["Name"] == "Generation")                           # only include rows for power generation data
    & (spec["Type"].isin(["DayAhead", "Actual"]))            # include both day-ahead forecasts and actual measured values
    & (spec["ProductionType"].isin([                          # restrict to the technologies of interest:
        'Hydro Water Reservoir',
        'Hydro Run-of-river and poundage',
        'Hydro Pumped Storage',
        'Wind Onshore',
        'Wind Offshore',
        'Solar',
        'DC Link',
        'AC Link'
    ]))
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )                           # limit to the Germany-Luxembourg bidding zone
    & (spec["MapTypeCode"] == "BZN")                         # ensure we’re looking at a bidding-zone level (not country/TSO)
]


# %% Download data for generation variable, 1 obs HOUR, and last months more than 1 OBS per hour and we need to fix
ds_generation = download_data(targets_generation, engine, 2019, 2025)
# Keep only the four columns we need for the time-series matrix
ds_generation = ds_generation[["DateTime", "Type", "ProductionType", "Value", 'MapCode']]
ds_generation = ds_generation.sort_values(by="DateTime")

ds_generation = ds_generation.pivot_table(
    index=["DateTime", "MapCode"],          # Keep MapCode as part of the index
    columns=["Type", "ProductionType"],     # Only pivot by Type and ProductionType
    values="Value"
).reset_index() 

# Define mapping for suffixes
suffix_map = {'Actual': 'A', 'DayAhead': 'DA'}

# Flatten the column multi-index and format the names
ds_generation.columns = [
    'time_utc' if col[0] == 'DateTime' else 
    'MapCode' if col[0] == 'MapCode' else 
    f"{col[1].replace(' ', '_').replace('-', '_')}_{suffix_map.get(col[0], col[0])}"
    for col in ds_generation.columns
]

# test obs per year
ds_generation_tmp = ds_generation.copy()
ds_generation_tmp['year'] = ds_generation_tmp['time_utc'].dt.year

# fix datetime to get the initial hours
ds_generation['time_utc'] = pd.to_datetime(ds_generation['time_utc']).dt.floor('H')
ds_generation = ds_generation.sort_values(['MapCode', 'time_utc']) 

# get group data per hour and mapcode
ds_generation_pl = pl.from_pandas(ds_generation)
value_cols = [col for col in ds_generation_pl.columns if col not in ["MapCode", "time_utc"]]

# Apply groupby and custom aggregation
ds_generation_pl = (

    ds_generation_pl
    .group_by(["MapCode", "time_utc"])
    .agg([
        pl.col(col).map_elements(custom_sum_polars).alias(col) for col in value_cols
    ])
    .sort(["MapCode", "time_utc"])
)


#%% check obs per year
results = []
for year in range(2010, 2026):
    date_range = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="H",
        tz="UTC"  # No DST applied
    )
    results.append({
        "Year": year,
        "Hourly_Observations": len(date_range),
        "Is_Leap_Year": len(date_range) == 8784
    })

ds = pd.DataFrame(results)

# %% ************** LOAD ******************************
# Download data for load variable, 1 OBS PER HOUR, but need to change per hour
targets_load = spec[
    (spec["Name"] == "Load")
    & (spec["Type"].isin(["DayAhead", "Actual"]))
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  # Put desired BZN here
    & (spec["MapTypeCode"] == "BZN")
]

ds_load = download_data(targets_load, engine,2019, 2025)
ds_load = ds_load[["DateTime", "Type", "Value", 'MapCode']]
ds_load = ds_load.pivot_table(
    index=["DateTime", "MapCode"], 
    columns=["Type"], 
    values="Value"
).reset_index()
ds_load.columns.name = None
ds_load = ds_load.sort_values(['MapCode', 'DateTime']) 
ds_load.columns=["time_utc",'MapCode', "Load_A", "Load_DA"]

# test obs per year
ds_load_tmp = ds_load.copy()
ds_load_tmp['year'] = ds_load_tmp['time_utc'].dt.year

# fix datetime to get the initial hours
ds_load['time_utc'] = pd.to_datetime(ds_load['time_utc']).dt.floor('H')
ds_load_pl = pl.from_pandas(ds_load)
value_cols = [col for col in ds_load_pl.columns if col not in ["MapCode", "time_utc"]]
# get group data per hour and mapcode
ds_load_pl = (
    ds_load_pl
    .group_by(["MapCode", "time_utc"])
    .agg([
        pl.col(col).map_elements(custom_sum_polars).alias(col) for col in value_cols
    ])
    .sort(["MapCode", "time_utc"])
)


# %% ************** PRICE ******************************
# Download Price data; 1 OBS PER HOUR and other dates we have >1 obs in the last days, but need to change per hour
targets_price = spec[
    (spec["Name"] == "Price")
    & (spec["Type"].isin(["DayAhead", "Actual"]))
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  
    & (spec["MapTypeCode"] == "BZN")
]
ds_price = download_data(targets_price, engine, 2019, 2025)
ds_price = ds_price[["DateTime", "Type", "Value", 'MapCode']] #here i only have 'DayAhead' type
ds_price = ds_price.pivot_table(
    index=["DateTime", "MapCode"], 
    columns=["Type"], 
    values="Value"
).reset_index()
ds_price.columns.name = None
ds_price = ds_price.sort_values(by=['MapCode', 'DateTime'])
ds_price.columns=["time_utc", 'MapCode',"Price"]

# test obs per year
ds_price_tmp = ds_load.copy()
ds_price_tmp['year'] = ds_price_tmp['time_utc'].dt.year

# fix datetime to get the initial hours
ds_price['time_utc'] = pd.to_datetime(ds_price['time_utc']).dt.floor('H')
# get group data per hour and mapcode
ds_price_pl = pl.from_pandas(ds_price)
value_cols = [col for col in ds_price_pl.columns if col not in ["MapCode", "time_utc"]]
# get group data per hour and mapcode
ds_price_pl = (
    ds_price_pl
    .group_by(["MapCode", "time_utc"])
    .agg([
        pl.col(col).map_elements(custom_mean_polars).alias(col) for col in value_cols
    ])
    .sort(["MapCode", "time_utc"])
)

# %% ************** FillingRateHydro ******************************
# Download FillingRateHydro data; 1 OBS PER WEEK
targets_FillingRateHydro = spec[
    (spec["Name"] == "FillingRateHydro")
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  
    & (spec["MapTypeCode"] == "BZN")
]

ds_FillingRateHydro = download_data(targets_FillingRateHydro, engine, 2019, 2025)
ds_FillingRateHydro = ds_FillingRateHydro[["DateTime", "Value", 'MapCode']]
ds_FillingRateHydro = ds_FillingRateHydro.sort_values(['MapCode', 'DateTime']) 
ds_FillingRateHydro.columns = ["time_utc", "FillingRateHydro", 'MapCode']

# test obs per year
ds_FillingRateHydro_tmp = ds_FillingRateHydro.copy()
ds_FillingRateHydro_tmp['year'] = ds_FillingRateHydro_tmp['time_utc'].dt.year

ds_FillingRateHydro_pl = pl.from_pandas(ds_FillingRateHydro)


# %% Download PhysicalFlow data; more than 1 OBS PER hour, >1 obs in the last days
targets_PhysicalFlow = spec[
    (spec["Name"] == "PhysicalFlow")
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  
    & (spec["MapTypeCode"] == "BZN")
]
ds_PhysicalFlow = download_data(targets_PhysicalFlow, engine, 2019,2025)

ds_PhysicalFlow = ds_PhysicalFlow[["DateTime", "Value", 'MapCode']]
ds_PhysicalFlow = ds_PhysicalFlow.sort_values(by=['MapCode', 'DateTime'])

ds_PhysicalFlow.columns=["time_utc",'PhysicalFlow' ,'MapCode']

# test obs per year
ds_PhysicalFlow_tmp = ds_PhysicalFlow.copy()
ds_PhysicalFlow_tmp['year'] = ds_PhysicalFlow_tmp['time_utc'].dt.year

# fix datetime to get the initial hours
ds_PhysicalFlow['time_utc'] = pd.to_datetime(ds_PhysicalFlow['time_utc']).dt.floor('H')
# get group data per hour and mapcode
ds_PhysicalFlow_pl = pl.from_pandas(ds_PhysicalFlow)
value_cols = [col for col in ds_PhysicalFlow_pl.columns if col not in ["MapCode", "time_utc"]]
# get group data per hour and mapcode
ds_PhysicalFlow_pl = (
    ds_PhysicalFlow_pl
    .group_by(["MapCode", "time_utc"])
    .agg([
        pl.col(col).map_elements(custom_sum_polars).alias(col) for col in value_cols
    ])
    .sort(["MapCode", "time_utc"])
)


# %% Download InstalledCapacity data; 1 OBS PER YEAR
targets_InstalledCapacity = spec[
    (spec["Name"] == "InstalledCapacity")
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  
    & (spec["MapTypeCode"] == "BZN")
]
ds_InstalledCapacity= download_data(targets_InstalledCapacity, engine, 2019, 2025)
ds_InstalledCapacity = ds_InstalledCapacity[["DateTime", "Type", "ProductionType", "Value", 'MapCode']]
ds_InstalledCapacity = ds_InstalledCapacity.sort_values(by="DateTime")
ds_InstalledCapacity = ds_InstalledCapacity.pivot_table(
    index="DateTime", columns=["ProductionType"], values="Value"
).reset_index()

# Rename columns
ds_InstalledCapacity.columns = [
    'time_utc' if col == 'DateTime' else f"{col.replace(' ', '_').replace('-', '_')}_InstCapac"
    for col in ds_InstalledCapacity.columns
]

ds_InstalledCapacity_pl = pl.from_pandas(ds_InstalledCapacity)


# %% Download Unavailability data; 1 OBS PER YEAR
targets_Unavailability = spec[
    (spec["Name"] == "Unavailability")
    & (spec["MapCode"].isin(["NO1", "NO2", "NO3", "NO4", "NO5"]) )  
    & (spec["MapTypeCode"] == "BZN")
]
ds_Unavailability = download_data(targets_Unavailability, engine,2019, 2025)
ds_Unavailability = ds_Unavailability[["DateTime", "Type", "Value", 'MapCode']]

ds_Unavailability = ds_Unavailability.pivot_table(index=["DateTime", "MapCode"], 
                                                  columns=["Type"], values='Value'
                                                  ).reset_index()
ds_Unavailability = ds_Unavailability.sort_values(by=['MapCode',"DateTime"])
ds_Unavailability.columns.name = None
ds_Unavailability['DateTime'] = pd.to_datetime(ds_Unavailability['DateTime']).dt.floor('H')
rename_map = {
    'DateTime': 'time_utc',
    'MapCode': 'MapCode',
    'Consumption': 'Cons_Unavailability',
    'Generation': 'Gen_Unavailability',
    'Production': 'Prod_Unavailability',
    'Transmission': 'Tran_Unavailability'
}
ds_Unavailability.rename(columns=rename_map, inplace=True)
ds_Unavailability['Total_Unavailability'] = ds_Unavailability.loc[:,'Cons_Unavailability':'Tran_Unavailability'].sum(axis=1, skipna=True).round(2)
# group data per hour and type
ds_Unavailability_pl = pl.from_pandas(ds_Unavailability)
value_cols = [col for col in ds_Unavailability_pl.columns if col not in ["MapCode", "time_utc"]]
# get group data per hour and mapcode
ds_Unavailability_pl = (
    ds_Unavailability_pl
    .group_by(["MapCode", "time_utc"])
    .agg([
        pl.col(col).map_elements(custom_sum_polars).alias(col) for col in value_cols
    ])
    .sort(["MapCode", "time_utc"])
)


#%% 
#######################################################################
#                   Data from DATASTREAM
#########################################################################
# Define MySQL connection parameters to extract data from DATASTREAM
db_user_ = "student"
db_password_ = "#q6a21I&OA5k"
host_ = "132.252.60.112"
port_ = 3306
dbname_ = "DATASTREAM"
engine_ = create_engine(
    f"mysql://{urllib.parse.quote_plus(db_user_)}:{urllib.parse.quote_plus(db_password_)}@{host_}:{port_}/{dbname_}"
)
query_ = "SELECT * FROM datastream"
datastream_table = pd.read_sql_query(query_, engine_)

# Check the available tables in engine
inspector_ = inspect(engine_)
tables_ = inspector_.get_table_names()
print(tables_)

# %% Select the variable of interest
#_fM_0i = i front month product 
#_fQ_0i = i front quarter product 
#_fY_0i = i front year product 
# EUA_spot is the spot market price for EUA
# The other variables are the exchange rates
# OiL and Coal are in USD
var = ['coal_fM_01',
 'gas_fM_01',
 'oil_fM_01',
 'EUA_fM_01',
 'USD_EUR']

# Select only rows with required variables
datastream_table = datastream_table[datastream_table["name"].isin(var)].copy()

# Convert 'Date' to a proper datetime, then filter from '2014-01-01' onward
datastream_table["Date"] = pd.to_datetime(datastream_table["Date"], errors="coerce")
datastream_table = datastream_table.sort_values(by="Date")
datastream_table = datastream_table[(datastream_table["Date"] >= "2019-01-01") &
                                    (datastream_table["Date"] < "2025-01-01")].copy()

# Drop the RIC column
if "RIC" in datastream_table.columns:
    datastream_table.drop(columns=["RIC"], inplace=True)

# change datastream_table to a pivot table
df_wide_streamtable = datastream_table.pivot(index="Date", columns="name", values="Value")
df_wide_streamtable = df_wide_streamtable.reset_index()

df_wide_streamtable['oil_fM_01_EUR'] = df_wide_streamtable['oil_fM_01'] / df_wide_streamtable['USD_EUR']
df_wide_streamtable['coal_fM_01_EUR'] = df_wide_streamtable['coal_fM_01'] / df_wide_streamtable['USD_EUR']
df_wide_streamtable.columns.name = None
# df_wide_streamtable['gas_fM_01_EUR'] = df_wide_streamtable['gas_fM_01'] / df_wide_streamtable['USD_EUR']
rename_map = {
    'Date': 'time_utc'
}
df_wide_streamtable.rename(columns=rename_map, inplace=True)
df_wide_streamtable_pl = pl.from_pandas(df_wide_streamtable)

#%% 
#######################################################################
#                   Data from DWD_MOSMIX/ api.open-meteo -- WEATHER DATA
#########################################################################
# Coordinates for NO1 to NO5 (representative cities)
regions = {
    "NO1": {"lat": 59.9139, "lon": 10.7522},
    "NO2": {"lat": 58.1467, "lon": 7.9956},
    "NO3": {"lat": 63.4305, "lon": 10.3951},
    "NO4": {"lat": 69.6496, "lon": 18.9560},
    "NO5": {"lat": 60.39299, "lon": 5.32415},
}

start_date = "2019-01-01"
end_date = "2024-12-31"

dfs = []

for region, coords in regions.items():
    lat = coords["lat"]
    lon = coords["lon"]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "shortwave_radiation",
            "windspeed_10m",
            "winddirection_10m",
            "pressure_msl",
            "relative_humidity_2m"
        ]),
        "timezone": "Europe/Oslo"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()["hourly"]

    df = pd.DataFrame(data)
    df["MapCode"] = region
    df.rename(columns={
        "temperature_2m": "Temp",
        "shortwave_radiation": "Solar",
        "windspeed_10m": "WindS",
        "winddirection_10m": "WindDir",
        "pressure_msl": "Press",
        "relative_humidity_2m": "Humid"
    }, inplace=True)
    df["time_utc"] = pd.to_datetime(df["time"])
    dfs.append(df)

weather_df = pd.concat(dfs)
weather_df = weather_df.sort_values(["MapCode", "time_utc"]).reset_index(drop=True)
weather_df = weather_df.drop(columns='time')
weather_df_pl = pl.from_pandas(weather_df)


# %%
# ***************************************** MERGE ALL DATASETS ************************************************

merged_data = (
    ds_price_pl
    .join(ds_load_pl, on=["MapCode", "time_utc"], how="left")
    .join(ds_generation_pl, on=["MapCode", "time_utc"], how="left")
)
