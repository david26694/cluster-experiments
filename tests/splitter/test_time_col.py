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
    "df,switchback_freq,n_splits",
    [
        ("date_df", "1D", 10),
        ("date_df", "2D", 5),
        ("date_df", "4D", 3),
        ("hourly_df", "H", 240),
        ("hourly_df", "2H", 120),
        ("hourly_df", "4H", 60),
        ("hourly_df", "6H", 40),
        ("hourly_df", "12H", 20),
        ("minute_df", "min", 14400),
        ("minute_df", "2min", 7200),
        ("minute_df", "4min", 3600),
        ("minute_df", "30min", 480),
        ("minute_df", "60min", 240),
        ("yearly_df", "W", 53),
        ("yearly_df", "M", 12),
    ],
)
@switchback_splitter_parametrize
def test_date_col(splitter, df, switchback_freq, n_splits, request):
    date_df = request.getfixturevalue(df)
    splitter = request.getfixturevalue(splitter)
    splitter.switch_frequency = switchback_freq
    time_col = splitter._get_time_col(date_df)
    assert time_col.dtype == "datetime64[ns]"
    assert time_col.nunique() == n_splits

    if "W" not in switchback_freq and "M" not in switchback_freq:
        pd.testing.assert_series_equal(
            time_col, date_df["time"].dt.floor(switchback_freq)
        )


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
