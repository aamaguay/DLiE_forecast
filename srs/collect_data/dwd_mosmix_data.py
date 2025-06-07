"""weather_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pull hourly re‑analysis / forecast archive from **api.open‑meteo.com** for the
Norwegian bidding zones NO1 … NO5 and reshape it so it lines up with the ENTSOE
series.

The helper mirrors the style of ``entsoe_data`` and ``datastream_data`` so the
three modules can be chained in a larger ETL.
"""

from typing import Dict, List, Optional
import pandas as pd
import polars as pl
import requests


###############################################################################
# Defaults
###############################################################################

# Coordinates of a representative city inside each bidding zone
DEFAULT_REGIONS: Dict[str, Dict[str, float]] = {
    "NO1": {"lat": 59.9139, "lon": 10.7522},   # Oslo
    "NO2": {"lat": 58.1467, "lon": 7.9956},    # Kristiansand
    "NO3": {"lat": 63.4305, "lon": 10.3951},   # Trondheim
    "NO4": {"lat": 69.6496, "lon": 18.9560},   # Tromsø
    "NO5": {"lat": 60.39299, "lon": 5.32415},  # Bergen
}

DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2024-12-31" 

###############################################################################
# Single‑region helper
###############################################################################

def _build_payload(lat: float, lon: float, start: str, end: str) -> Dict[str, str]:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(
            [
                "temperature_2m",
                "shortwave_radiation",
                "windspeed_10m",
                "winddirection_10m",
                "pressure_msl",
                "relative_humidity_2m",
            ]
        ),
        "timezone": "Europe/Oslo",  # UTC+1 / UTC+2 DST – consistent with ENTSOE MapCodes
    }


def fetch_region_weather(
    region_code: str,
    lat: float,
    lon: float,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Query the Open‑Meteo *archive* endpoint for one region and return a tidy
    *pandas* frame.

    Parameters
    ----------
    region_code : str
        Label to store in the resulting ``MapCode`` column (e.g. "NO3").
    lat, lon : float
        Coordinates.
    start, end : str (YYYY‑MM‑DD)
        Inclusive date range understood by the API.
    session : requests.Session | None
        Optional pre‑configured session for connection‑pool reuse.

    Returns
    -------
    pandas.DataFrame
        Hourly observations with columns::

            time_utc  Temp  Solar  WindS  WindDir  Press  Humid  MapCode
    """

    sess = session or requests
    url = "https://archive-api.open-meteo.com/v1/archive"
    resp = sess.get(url, params=_build_payload(lat, lon, start, end), timeout=60)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)

    # Rename and enrich
    df.rename(
        columns={
            "time": "time_utc",
            "temperature_2m": "Temp",
            "shortwave_radiation": "Solar",
            "windspeed_10m": "WindS",
            "winddirection_10m": "WindDir",
            "pressure_msl": "Press",
            "relative_humidity_2m": "Humid",
        },
        inplace=True,
    )
    df["MapCode"] = region_code
    df["time_utc"] = pd.to_datetime(df["time_utc"])  # keep tz‑naïve (UTC implied)

    return df

###############################################################################
# Multi‑region orchestrator
###############################################################################

def prepare_weather(
    regions: Dict[str, Dict[str, float]] | None = None,
    start: str = DEFAULT_START_DATE,
    end: str = DEFAULT_END_DATE,
) -> pl.DataFrame:
    """Download weather for several MapCodes and return a *polars* frame."""

    coords = regions or DEFAULT_REGIONS

    session = requests.Session()  # connection pool for speed
    frames: List[pd.DataFrame] = []

    for code, loc in coords.items():
        frames.append(
            fetch_region_weather(
                region_code=code,
                lat=loc["lat"],
                lon=loc["lon"],
                start=start,
                end=end,
                session=session,
            )
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["MapCode", "time_utc"], ignore_index=True)
    )

    # Return in Polars for consistency with other modules
    return pl.from_pandas(combined)