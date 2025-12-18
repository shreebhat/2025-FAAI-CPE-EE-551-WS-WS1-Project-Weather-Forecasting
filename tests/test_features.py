import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))

from weather.features import build_features, stream_days


# Unit tests for feature engineering functions.
# Focuses on correctness, shape, and expected columns.


def _df_ok(n=120):
    return pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-01", periods=n, freq="D"),
            "DailyAverageDryBulbTemperature": [float((i * 3) % 40) for i in range(n)],
        }
    )


def test_build_features_columns_exist():
    df = _df_ok()
    out = build_features(df)

    assert "DailyAverageDryBulbTemperature_lag_1" in out.columns
    assert "DailyAverageDryBulbTemperature_rollmean_7" in out.columns
    assert "month" in out.columns
    assert "doy_sin" in out.columns
    assert "doy_cos" in out.columns


def test_build_features_drops_rows():
    df = _df_ok(80)
    out = build_features(df, lags=(1, 7), windows=(14,))
    assert len(out) < len(df)


def test_stream_days_order():
    df = _df_ok(40).sample(frac=1.0, random_state=42)
    items = list(stream_days(df))
    assert items[0][0] <= items[1][0]


def test_stream_days_count():
    df = _df_ok(30)
    items = list(stream_days(df))
    assert len(items) == len(df)
