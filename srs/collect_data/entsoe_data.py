import pandas as pd
import polars as pl
from typing import List, Optional, Dict
from sqlalchemy.engine import Engine
import urllib.parse
from sqlalchemy import create_engine, inspect

# Default MySQL connection parameters for ENTSOE database
ENTSOE_DB = {
    "user": "student",
    "password": "#q6a21I&OA5k",
    "host": "132.252.60.112",
    "port": 3306,
    "dbname": "ENTSOE"
}

# default configuration
DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2025
DEFAULT_MAPCODES = ["NO1", "NO2", "NO3", "NO4", "NO5"]

# running
def create_entsoe_engine(config: dict = None):
    """
    Create a SQLAlchemy engine for the ENTSOE MySQL database.

    Parameters:
    - config: Optional dict with keys 'user', 'password', 'host', 'port', 'dbname'.
              If None, uses the default ENTSOE_DB.

    Returns:
    - SQLAlchemy Engine
    """
    cfg = config or ENTSOE_DB
    url = (
        f"mysql+pymysql://"
        f"{urllib.parse.quote_plus(cfg['user'])}:"
        f"{urllib.parse.quote_plus(cfg['password'])}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )
    return create_engine(url)

# running
def get_tables(engine):
    """
    List all table names in the connected database.

    Parameters:
    - engine: SQLAlchemy Engine

    Returns:
    - List of table names (List[str])
    """
    inspector = inspect(engine)
    return inspector.get_table_names()

# running
def get_spec(engine) -> pd.DataFrame:
    """
    Load the full 'spec' table into a pandas DataFrame.

    Parameters:
    - engine: SQLAlchemy Engine

    Returns:
    - DataFrame with all rows from 'spec'
    """
    query = "SELECT * FROM spec"
    return pd.read_sql_query(query, engine)

# running
def get_market_divisions(spec_df: pd.DataFrame):
    """
    Return unique market division codes (MapTypeCode) in spec.
    """
    return spec_df.MapTypeCode.unique()

# running
def get_map_codes(spec_df: pd.DataFrame):
    """
    Return all unique MapCode values in spec.
    """
    return spec_df.MapCode.unique()

# running
def get_map_codes_starting_with(spec_df: pd.DataFrame, prefix: str):
    """
    Return MapCode values starting with the given prefix.
    """
    mask = spec_df.MapCode.fillna('').str.startswith(prefix)
    return spec_df.MapCode[mask].unique()

# running
def get_resolution_codes(spec_df: pd.DataFrame):
    """
    Return unique ResolutionCode values in spec (e.g., 'PT60M').
    """
    return spec_df.ResolutionCode.unique()

#
def download_data(targets: pd.DataFrame, 
                  engine, 
                  start_year: int = DEFAULT_START_YEAR,
                  end_year: int = DEFAULT_END_YEAR,) -> pd.DataFrame:
    """
    Download time-series values for the given TimeSeriesIDs and merge with spec metadata.

    Parameters:
    - targets: DataFrame with a column 'TimeSeriesID'
    - engine: SQLAlchemy Engine
    - yr: inclusive start year (e.g., 2014)
    - yr_end: exclusive end year (e.g., 2025)

    Returns:
    - DataFrame with merged 'values' and 'targets'
    """
    ids = ", ".join(map(str, targets['TimeSeriesID']))
    values_query = f"""
    SELECT *
    FROM vals
    WHERE TimeSeriesID IN ({ids})
      AND YEAR(`DateTime`) >= '{start_year}'
      AND YEAR(`DateTime`) < '{end_year}'
    """
    values = pd.read_sql_query(values_query, engine)
    merged = pd.merge(values, targets, on='TimeSeriesID')
    return merged

#
def group_hourly(
    df: pd.DataFrame,
    key_cols: List[str],
    value_cols: List[str],
    agg_func: str = 'sum',
) -> pl.DataFrame:
    """
    Floor datetime to hour, group by key_cols, and aggregate value_cols.

    Parameters
    ----------
    df : pandas.DataFrame
    key_cols : list of str
    value_cols : list of str
    agg_func : {'sum', 'mean'}

    Returns
    -------
    polars.DataFrame
    """
    df['time_utc'] = pd.to_datetime(df['time_utc']).dt.floor('h')
    pl_df = pl.from_pandas(df)
    if agg_func == 'sum':
        agg = [pl.col(c).map_elements(lambda s: None if s.null_count()==len(s) else s.sum(),
                                      return_dtype=float).alias(c)
               for c in value_cols]
    else:
        agg = [pl.col(c).map_elements(lambda s: None if s.null_count()==len(s) else s.mean(),
                                      return_dtype=float).alias(c)
               for c in value_cols]
    return (
        pl_df
        .group_by(key_cols)
        .agg(agg)
        .sort(key_cols)
    )

#
def prepare_generation(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare generation data for given mapcodes.
    """
    prod_types = [
        'Hydro Water Reservoir', 'Hydro Run-of-river and poundage',
        'Hydro Pumped Storage', 'Wind Onshore', 'Wind Offshore',
        'Solar', 'DC Link', 'AC Link'
    ]
    targets = spec[
        (spec['Name']=='Generation') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        (spec['ProductionType'].isin(prod_types)) &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','ProductionType','Value','MapCode']]
    df = df.sort_values('DateTime')
    pivot = df.pivot_table(
        index=['DateTime','MapCode'],
        columns=['Type','ProductionType'],
        values='Value'
    ).reset_index()
    # flatten column names
    suffix = {'Actual':'A','DayAhead':'DA'}
    pivot.columns = [
        'time_utc' if t=='DateTime' else
        'MapCode' if t=='MapCode' else
        f"{prod.replace(' ','_')}_{suffix.get(t,t)}"
        for (t,prod) in pivot.columns
    ]
    return group_hourly(
        pivot, key_cols=['MapCode','time_utc'],
        value_cols=[c for c in pivot.columns if c not in ['MapCode','time_utc']],
        agg_func='sum'
    )

#
def prepare_load(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare load data.
    """
    targets = spec[
        (spec['Name']=='Load') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','Value','MapCode']]
    pivot = df.pivot_table(
        index=['DateTime','MapCode'],
        columns=['Type'],
        values='Value'
    ).reset_index()
    pivot.columns = ['time_utc','MapCode','Load_A','Load_DA']
    return group_hourly(
        pivot, ['MapCode','time_utc'], ['Load_A','Load_DA'], agg_func='sum'
    )

#
def prepare_price(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare price data.
    """
    targets = spec[
        (spec['Name']=='Price') &
        (spec['Type'].isin(['DayAhead','Actual'])) &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Type','Value','MapCode']]
    pivot = df.pivot_table(
        index=['DateTime','MapCode'],
        columns=['Type'],
        values='Value'
    ).reset_index()
    pivot.columns = ['time_utc','MapCode','Price']
    return group_hourly(
        pivot, ['MapCode','time_utc'], ['Price'], agg_func='mean'
    )

#
def prepare_filling_rate_hydro(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download weekly filling rate hydro data.
    """
    targets = spec[
        (spec['Name']=='FillingRateHydro') &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Value','MapCode']].sort_values(['MapCode','DateTime'])
    df.columns = ['time_utc','FillingRateHydro','MapCode']
    return pl.from_pandas(df)

#
def prepare_physical_flow(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare physical flow data.
    """
    targets = spec[
        (spec['Name']=='PhysicalFlow') &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','Value','MapCode']].sort_values(['MapCode','DateTime'])
    df.columns = ['time_utc','PhysicalFlow','MapCode']
    return group_hourly(
        df, ['MapCode','time_utc'], ['PhysicalFlow'], agg_func='sum'
    )

#
def prepare_installed_capacity(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> pl.DataFrame:
    """
    Download and prepare annual installed capacity.
    """
    targets = spec[
        (spec['Name']=='InstalledCapacity') &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df[['DateTime','ProductionType','Value']].sort_values('DateTime')
    pivot = df.pivot_table(
        index='DateTime',
        columns='ProductionType',
        values='Value'
    ).reset_index()
    pivot.columns = [
        'time_utc' if c=='DateTime' else f"{c.replace(' ','_')}_InstCapac"
        for c in pivot.columns
    ]
    return pl.from_pandas(pivot)

#
def prepare_unavailability(
    spec: pd.DataFrame,
    engine: Engine,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    mapcodes: List[str] = DEFAULT_MAPCODES,
) -> pl.DataFrame:
    """
    Download and prepare unavailability data.
    """
    targets = spec[
        (spec['Name']=='Unavailability') &
        (spec['MapCode'].isin(mapcodes)) &
        (spec['MapTypeCode']=='BZN')
    ]
    df = download_data(targets, engine, start_year, end_year)
    df = df.pivot_table(
        index=['DateTime','MapCode'],
        columns='Type',
        values='Value'
    ).reset_index()
    df.columns.name = None
    df['time_utc'] = pd.to_datetime(df['DateTime']).dt.floor('h')
    df = df.rename(columns={
        'Consumption': 'Cons_Unavailability',
        'Generation':  'Gen_Unavailability',
        'Production':  'Prod_Unavailability',
        'Transmission':'Tran_Unavailability'
    })
    df['Total_Unavailability'] = df.filter(regex='_Unavailability$')\
        .sum(axis=1, skipna=True).round(2)
    return group_hourly(
        df, ['MapCode','time_utc'],
        [c for c in df.columns if c not in ['MapCode','time_utc','DateTime']],
        agg_func='sum'
    )
