import pandas as pd
import pytest

from weather.data import WeatherStation, DataValidationError


# Unit tests for WeatherStation data loading and validation.
# These tests ensure that bad input fails early and clean data loads correctly.


def _df_ok(n=80):
    return pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-01", periods=n, freq="D"),
            "DailyAverageDryBulbTemperature": [float(i % 50) for i in range(n)],
        }
    )


def test_load_success(tmp_path):
    p = tmp_path / "x.csv"
    _df_ok().to_csv(p, index=False)

    ws = WeatherStation(p)
    assert ws.df is not None
    assert len(ws.df) >= 50


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        WeatherStation(tmp_path / "nope.csv")


def test_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"DATE": pd.date_range("2020-01-01", periods=100, freq="D")}).to_csv(p, index=False)

    with pytest.raises(DataValidationError):
        WeatherStation(p)


def test_date_sorted(tmp_path):
    df = _df_ok(120).sample(frac=1.0, random_state=0)
    p = tmp_path / "shuf.csv"
    df.to_csv(p, index=False)

    ws = WeatherStation(p)
    assert ws.df["DATE"].is_monotonic_increasing


def test_str_has_rows(tmp_path):
    p = tmp_path / "x.csv"
    _df_ok().to_csv(p, index=False)

    ws = WeatherStation(p, station_name="Albany")
    s = str(ws)
    assert "WeatherStation" in s
    assert "rows=" in s
