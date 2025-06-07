"""merge_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities that combine the ENTSOE market data (price, load, generation) with
fuel‑/FX‑related variables from *datastream_data* into a single modelling table
and optionally persist the result to ``.csv``.
"""

from pathlib import Path
from typing import List, Optional
import pyarrow
import pandas as pd
import polars as pl

def merge_datasets(
    price: pl.DataFrame,
    load: pl.DataFrame,
    generation: pl.DataFrame,
    fuels: pl.DataFrame,
    join_type: str = "left",
) -> pl.DataFrame:
    
    """Step‑wise join that mirrors the tutor's notebook logic.

    Parameters
    ----------
    price, load, generation : pl.DataFrame
        Outputs of ``entsoe_data.prepare_*`` helpers – **must** share the pair
        ``(MapCode, time_utc)``.
    fuels : pl.DataFrame
        Output of ``datastream_data.prepare_datastream`` (no MapCode column).
    join_type : str
        Passed straight to ``DataFrame.join`` (typically "left" or "inner").

    Returns
    -------
    pl.DataFrame
    """

    merged = (
        price
        .join(load, on=["MapCode", "time_utc"], how=join_type)
        .join(generation, on=["MapCode", "time_utc"], how=join_type)
    )

    merged = merged.join(fuels, on=["time_utc"], how=join_type)

    return merged

###############################################################################
# Training‑ready slice
###############################################################################

def _ffill_weekend_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Forward‑fill *within each MapCode* – replicates tutor behaviour."""
    df[columns] = df.groupby("MapCode")[columns].transform("ffill")
    return df

_RENAME_MAP = {
    "Wind_Onshore_DA": "WindOn_DA",
    "Wind_Offshore_DA": "WindOff_DA",
    "coal_fM_01_EUR": "Coal",
    "gas_fM_01": "NGas",
    "oil_fM_01_EUR": "Oil",
    "EUA_fM_01": "EUA",
}

_REGRESSOR_COLUMNS = [
    "time_utc",
    "Price",
    "Load_A",
    "Load_DA",
    "WindOn_DA",
    "WindOff_DA",
    "Solar_DA",
    "Coal",
    "NGas",
    "Oil",
    "EUA",
]

def build_training_dataset(
    merged: pl.DataFrame,
    mapcode: Optional[str] = None,
    save_csv: Optional[str | Path] = None,
    fill_weekends: bool = True,
) -> pd.DataFrame:
    
    """Create the modelling table used by the tutor.

    Parameters
    ----------
    merged : pl.DataFrame
        Output of :func:`merge_datasets`.
    mapcode : str | None
        If given, keep only that bidding zone (e.g. "NO1"). If ``None`` keep
        all zones.
    save_csv : str | pathlib.Path | None
        If provided, the final *pandas* DataFrame is written to this path.
    fill_weekends : bool
        Apply the forward‑fill step the tutor used for Saturday/Sunday gaps in
        fuel/FX variables.

    Returns
    -------
    pandas.DataFrame
    """

    df = merged.to_pandas()

    if fill_weekends:
        _fill_cols = [
            "EUA_fM_01",
            "USD_EUR",
            "coal_fM_01",
            "gas_fM_01",
            "oil_fM_01",
            "oil_fM_01_EUR",
            "coal_fM_01_EUR",
            "Wind_Offshore_DA",
            "Wind_Onshore_DA",
        ]
        df = _ffill_weekend_values(df, _fill_cols)

    # Rename & select modelling columns
    df = df.rename(columns=_RENAME_MAP)

    if mapcode is not None:
        df = df[df["MapCode"] == mapcode]
        df = df.drop(columns="MapCode")

    df = df[_REGRESSOR_COLUMNS]

    if save_csv is not None:
        Path(save_csv).expanduser().with_suffix(".csv")
        df.to_csv(save_csv, index=False)

    return df