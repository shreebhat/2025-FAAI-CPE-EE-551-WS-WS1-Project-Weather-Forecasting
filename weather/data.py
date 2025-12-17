"""
Load + clean NOAA daily weather CSV for one station.

This handles basic validation, cleanup, and sanity checks so the rest
of the pipeline can assume the data is usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


class DataValidationError(Exception):
    """
    Raised when the input CSV exists but is unusable
    -missing columns, bad dates, empty after cleaning, etc
    """
    pass

@dataclass
class WeatherStation:
    # Represents a single weather station and its daily time-series data.
    # Handles all validation and cleaning so downstream code can assume
    # the dataframe is already safe to use.

    csv_path: Path | str
    station_name: str = "Albany (Representative Northeast Station)"
    date_col: str = "DATE"
    target_col: str = "DailyAverageDryBulbTemperature"
    df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        raw = self._load_csv()
        self.df = self._clean(raw)

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing file: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise DataValidationError(f"Could not read CSV: {e}") from e

        if df.empty:
            raise DataValidationError("CSV is empty.")

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        need = [self.date_col, self.target_col]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing columns: {missing}")

        out = df.copy()

        # parse date (NOAA uses timestamps at 23:59 sometimes)
        out[self.date_col] = pd.to_datetime(out[self.date_col], errors="coerce")
        if out[self.date_col].isna().all():
            raise DataValidationError(f"Bad dates in '{self.date_col}'")

        # NOAA timestamps sometimes include time; normalize to daily
        out[self.date_col] = out[self.date_col].dt.normalize()

        # Convert weird NOAA values (T, M, s) to NaN safely
        out[self.target_col] = pd.to_numeric(out[self.target_col], errors="coerce")
        if out[self.target_col].isna().all():
            raise DataValidationError(f"Bad numeric values in '{self.target_col}'")

        # drop rows missing date/target
        out = out.dropna(subset=[self.date_col, self.target_col])

        # keep one row per day
        out = out.sort_values(self.date_col).drop_duplicates(subset=[self.date_col], keep="last")
        out = out.reset_index(drop=True)

        if len(out) < 50:
            raise DataValidationError(f"Not enough daily rows after cleaning ({len(out)}).")

        return out

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.df is None or self.df.empty:
            raise DataValidationError("No data loaded.")
        s = self.df[self.date_col]
        return s.min(), s.max()

    def temp_stats(self) -> dict[str, float]:
        if self.df is None or self.df.empty:
            raise DataValidationError("No data loaded.")
        s = self.df[self.target_col]
        return {
            "count": float(s.count()),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "min": float(s.min()),
            "max": float(s.max()),
        }

    def __str__(self) -> str:
        if self.df is None or self.df.empty:
            return f"WeatherStation(name='{self.station_name}', rows=0)"

        a, b = self.date_range()
        stats = self.temp_stats()
        return (
            f"WeatherStation(name='{self.station_name}', rows={len(self.df)}, "
            f"range={a.date()}..{b.date()}, mean={stats['mean']:.2f})"
        )