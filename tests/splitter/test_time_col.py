import numpy as np
import pandas as pd
import pytest

from tests.splitter.conftest import switchback_splitter_parametrize


@pytest.fixture
def yearly_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", "2021-12-31", freq="1D"),
            "y": np.random.randn(365),
        }
    )


@pytest.fixture
def hourly_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", "2021-01-10 23:00", freq="1H"),
            "y": np.random.randn(10 * 24),  # 10 days, 24 hours per day, 1 hour per row
        }
    )


@pytest.fixture
def minute_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", "2021-01-10 23:59", freq="1T"),
            "y": np.random.randn(
                10 * 24 * 60
            ),  # 10 days, 24 hours per day, 1 hour per row
        }
    )


@pytest.mark.parametrize(
    "switchback_freq,n_splits",
    [("1D", 10), ("2D", 5), ("4D", 3)],
)
@switchback_splitter_parametrize
def test_date_col(splitter, date_df, switchback_freq, n_splits, request):
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(date_df)
    assert time_col.dtype == "datetime64[ns]"
    pd.testing.assert_series_equal(time_col, date_df["time"].dt.floor(switchback_freq))
    assert time_col.nunique() == n_splits


@pytest.mark.parametrize(
    "switchback_freq,day_of_week",
    [
        ("W-MON", "Tuesday"),
        ("W-TUE", "Wednesday"),
        ("W-SUN", "Monday"),
        ("W", "Monday"),
    ],
)
@switchback_splitter_parametrize
def test_week_col_date(splitter, date_df, switchback_freq, day_of_week, request):
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(date_df)
    assert time_col.dtype == "datetime64[ns]"
    pd.testing.assert_series_equal(
        time_col, date_df["time"].dt.to_period(switchback_freq).dt.start_time
    )
    assert time_col.nunique() == 2
    # Assert that the first day of the week is correct
    assert (
        time_col.dt.day_name().iloc[0] == day_of_week
    ), f"{switchback_freq} failed, day_of_week is {time_col.dt.day_name().iloc[0]}"


@pytest.mark.parametrize(
    "switchback_freq,n_splits",
    [
        ("M", 12),
        ("W", 53),
    ],
)
@switchback_splitter_parametrize
def test_week_col(splitter, yearly_df, switchback_freq, n_splits, request):
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(yearly_df)
    assert time_col.dtype == "datetime64[ns]"
    assert time_col.nunique() == n_splits, f"nunique is {time_col.nunique()}"


@pytest.mark.parametrize(
    "switchback_freq,n_splits",
    [
        ("H", 240),
        ("2H", 120),
        ("4H", 60),
        ("6H", 40),
        ("12H", 20),
    ],
)
@switchback_splitter_parametrize
def test_hourly_col(splitter, hourly_df, switchback_freq, n_splits, request):
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(hourly_df)
    assert time_col.dtype == "datetime64[ns]"
    assert time_col.nunique() == n_splits, f"nunique is {time_col.nunique()}"


@pytest.mark.parametrize(
    "switchback_freq,n_splits",
    [
        ("min", 14400),
        ("2min", 7200),
        ("4min", 3600),
        ("30min", 480),
        ("60min", 240),
    ],
)
@switchback_splitter_parametrize
def test_minute_col(splitter, minute_df, switchback_freq, n_splits, request):
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(minute_df)
    assert time_col.dtype == "datetime64[ns]"
    assert time_col.nunique() == n_splits, f"nunique is {time_col.nunique()}"
