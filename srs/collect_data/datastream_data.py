"""datastream_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Helpers for downloading and reshaping the *DATASTREAM* MySQL feed
so that it matches the hourly granularity used by the ENTSOE helpers.

Typical usage
-------------
>>> import datastream_data as ds
>>> eng = ds.create_datastream_engine()
>>> prices = ds.prepare_datastream(eng)

The public surface mimics ``entsoe_data`` so the two modules can be
used side‑by‑side in a larger data‑pipeline.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import polars as pl
import urllib.parse
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

###############################################################################
# Database configuration
###############################################################################

DATASTREAM_DB: Dict[str, str | int] = {
    "user": "student",
    "password": "#q6a21I&OA5k",
    "host": "132.252.60.112",
    "port": 3306,
    "dbname": "DATASTREAM",
}

# Default parameters for the prep helper
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2025-01-01" 
DEFAULT_VARS = [
    "coal_fM_01",  # front‑month ICE Rotterdam coal (USD)
    "gas_fM_01",   # front‑month TTF gas (EUR) – already EUR‑denominated
    "oil_fM_01",   # ICE Brent (USD)
    "EUA_fM_01",   # EUA allowance front‑month (EUR)
    "USD_EUR",     # FX rate USD → EUR (quote: 1 USD = x EUR)
]

###############################################################################
# Engine / Meta helpers
###############################################################################

def _make_url(cfg: Dict[str, str | int]) -> str:
    """Return a SQLAlchemy URL for *pymysql*."""
    return (
        "mysql+pymysql://"
        f"{urllib.parse.quote_plus(str(cfg['user']))}:"
        f"{urllib.parse.quote_plus(str(cfg['password']))}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )


def create_datastream_engine(cfg: Optional[Dict[str, str | int]] = None) -> Engine:
    """Instantiate a SQLAlchemy engine for the DATASTREAM schema."""
    return create_engine(_make_url(cfg or DATASTREAM_DB))


def get_tables(engine: Engine) -> List[str]:
    """List tables available in the connected database."""
    return inspect(engine).get_table_names()

###############################################################################
# ETL helper
###############################################################################

def prepare_datastream(
    engine: Engine,
    variables: List[str] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pl.DataFrame:
    """Download daily market data and upsample to hourly.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Active connection to the DATASTREAM database.
    variables : list[str] | None
        Which *name* values to keep.  ``None`` → use ``DEFAULT_VARS``.
    start, end : str (YYYY‑MM‑DD)
        Inclusive lower / exclusive upper date filter.

    Returns
    -------
    pl.DataFrame
        Hourly‑grained wide table.  Column ``time_utc`` is timezone‑naive
        but refers to UTC.
    """
    vars_to_use = variables or DEFAULT_VARS

    # ------------------------------------------------------------------
    # Fetch raw table (single source of truth: "datastream")
    # ------------------------------------------------------------------
    raw = pd.read_sql_query("SELECT * FROM datastream", engine)

    # ------------------------------------------------------------------
    # Filtering and cleaning
    # ------------------------------------------------------------------
    raw = raw[raw["name"].isin(vars_to_use)].copy()

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    mask = (raw["Date"] >= start) & (raw["Date"] < end)
    raw = raw.loc[mask]
    raw = raw.sort_values("Date")

    # Optional column present in some dumps
    if "RIC" in raw.columns:
        raw = raw.drop(columns="RIC")

    # ------------------------------------------------------------------
    # Pivot into wide, one row per calendar day
    # ------------------------------------------------------------------
    wide = raw.pivot(index="Date", columns="name", values="Value").reset_index()

    # ------------------------------------------------------------------
    # Currency conversions (USD‑quoted commodities → EUR)
    # ------------------------------------------------------------------
    if {"oil_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["oil_fM_01_EUR"] = wide["oil_fM_01"] / wide["USD_EUR"]
    if {"coal_fM_01", "USD_EUR"}.issubset(wide.columns):
        wide["coal_fM_01_EUR"] = wide["coal_fM_01"] / wide["USD_EUR"]
    # If you need gas FX conversion uncomment similar logic here

    wide.columns.name = None
    wide = wide.rename(columns={"Date": "time_utc"})

    # ------------------------------------------------------------------
    # Upsample: replicate each daily row 24× to create hourly data
    # ------------------------------------------------------------------
    n = len(wide)
    wide = wide.loc[wide.index.repeat(24)].copy()

    hours = np.tile(np.arange(24, dtype="timedelta64[h]"), n)
    wide["time_utc"] = pd.to_datetime(wide["time_utc"]) + hours

    # ------------------------------------------------------------------
    # Return as Polars for parity with entsoe_data helpers
    # ------------------------------------------------------------------
    return pl.from_pandas(wide)

