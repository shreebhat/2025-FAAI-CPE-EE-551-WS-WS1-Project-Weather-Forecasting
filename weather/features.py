"""
Feature engineering for daily temperature forecasting.

Creates lag features, rolling statistics, and seasonal features
from a cleaned daily weather time series.
"""


from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def build_features(
    # Builds supervised learning features from a daily time series.
    # All rolling features are shifted to prevent data leakage.

    df: pd.DataFrame,
    date_col: str = "DATE",
    target_col: str = "DailyAverageDryBulbTemperature",
    lags: tuple[int, ...] = (1, 2, 3, 7),
    windows: tuple[int, ...] = (3, 7, 14),
) -> pd.DataFrame:
    
    #Returns a df with features + the original target col.
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("df missing date/target columns")

    out = df.copy()
    out = out.sort_values(date_col).reset_index(drop=True)

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().any():
        raise ValueError("Bad dates found during feature build")

    # lag feature capture recent temp history
    lag_cols = [f"{target_col}_lag_{k}" for k in lags]  # list comp requirement
    for k, name in zip(lags, lag_cols):
        out[name] = out[target_col].shift(k)

    # rolling (shift 1 to avoid leakage)
    for w in windows:
        base = out[target_col].shift(1)
        out[f"{target_col}_rollmean_{w}"] = base.rolling(w).mean()
        out[f"{target_col}_rollstd_{w}"] = base.rolling(w).std()

    # seasonal stuff
    out["month"] = out[date_col].dt.month
    out["dayofyear"] = out[date_col].dt.dayofyear
    out["weekday"] = out[date_col].dt.weekday

    # map/lambda requirement
    angles = list(map(lambda d: 2.0 * np.pi * (d / 365.25), out["dayofyear"]))
    out["doy_sin"] = np.sin(angles)
    out["doy_cos"] = np.cos(angles)

    out = out.dropna().reset_index(drop=True)
    if len(out) < 20:
        raise ValueError("Too few rows after features; reduce lags/windows or check data")

    return out


def stream_days(df: pd.DataFrame, date_col: str = "DATE") -> Iterator[tuple[pd.Timestamp, pd.Series]]:
    
    #Generator: yields (date, row) in order.
    
    ordered = df.sort_values(date_col)
    for _, row in ordered.iterrows():
        yield pd.to_datetime(row[date_col]), row
