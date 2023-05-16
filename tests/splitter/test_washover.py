from dataclasses import dataclass
from datetime import timedelta

import pytest

from cluster_experiments import SwitchbackSplitter
from cluster_experiments.washover import ConstantWashover, EmptyWashover


@pytest.mark.parametrize("minutes, n_rows", [(30, 2), (10, 4), (15, 3)])
def test_constant_washover_base(minutes, n_rows, washover_base_df):
    out_df = ConstantWashover(washover_time_delta=timedelta(minutes=minutes)).washover(
        df=washover_base_df,
        truncated_time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
        original_time_col="original___time",
    )

    assert len(out_df) == n_rows
    assert (out_df["original___time"].dt.minute > minutes).all()


@pytest.mark.parametrize(
    "minutes, n_rows, df",
    [
        (30, 4, "washover_df_no_switch"),
        (30, 4 + 4, "washover_df_multi_city"),
    ],
)
def test_constant_washover_no_switch(minutes, n_rows, df, request):
    washover_df = request.getfixturevalue(df)

    out_df = ConstantWashover(washover_time_delta=timedelta(minutes=minutes)).washover(
        df=washover_df,
        truncated_time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
    )
    assert len(out_df) == n_rows
    if df == "washover_df_no_switch":
        # Check that, after 2022-01-01 02:00:00, we keep all the rows of the original
        # dataframe
        assert washover_df.query("time >= '2022-01-01 02:00:00'").equals(
            out_df.query("time >= '2022-01-01 02:00:00'")
        )
        # Check that, after 2022-01-01 01:00:00, we don't have the same rows as the
        # original dataframe
        assert not washover_df.query("time >= '2022-01-01 01:00:00'").equals(
            out_df.query("time >= '2022-01-01 01:00:00'")
        )


@pytest.mark.parametrize(
    "minutes, n",
    [
        (15, 10000),
    ],
)
def test_constant_washover_split(minutes, n, washover_split_df):
    washover = ConstantWashover(washover_time_delta=timedelta(minutes=minutes))

    splitter = SwitchbackSplitter(
        washover=washover,
        time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
        switch_frequency="30T",
    )

    out_df = splitter.assign_treatment_df(df=washover_split_df)

    # Assert A and B in out_df
    assert set(out_df["treatment"].unique()) == {"A", "B"}

    # We need to have less than 10000 rows
    assert len(out_df) < n

    # We need to have more than 5000 rows (this is because ABB doesn't do washover on the second split)
    assert len(out_df) > n / 2


@pytest.mark.parametrize(
    "minutes, n",
    [
        (15, 1000),
    ],
)
def test_no_washover_split(minutes, n, washover_split_df):
    washover = EmptyWashover()

    splitter = SwitchbackSplitter(
        washover=washover,
        time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
        switch_frequency="30T",
    )

    out_df = splitter.assign_treatment_df(df=washover_split_df)

    # Assert A and B in out_df
    assert set(out_df["treatment"].unique()) == {"A", "B"}

    # We need to have exactly 1000 rows
    assert len(out_df) == n


@pytest.mark.parametrize(
    "minutes, n_rows, df",
    [
        (30, 4, "washover_df_no_switch"),
        (30, 4 + 4, "washover_df_multi_city"),
    ],
)
def test_constant_washover_no_switch_instantiated_int(minutes, n_rows, df, request):
    washover_df = request.getfixturevalue(df)

    @dataclass
    class Cfg:
        washover_time_delta: int

    cw = ConstantWashover.from_config(Cfg(minutes))
    out_df = cw.washover(
        df=washover_df,
        truncated_time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
    )
    assert len(out_df) == n_rows
    if df == "washover_df_no_switch":
        # Check that, after 2022-01-01 02:00:00, we keep all the rows of the original
        # dataframe
        assert washover_df.query("time >= '2022-01-01 02:00:00'").equals(
            out_df.query("time >= '2022-01-01 02:00:00'")
        )
        # Check that, after 2022-01-01 01:00:00, we don't have the same rows as the
        # original dataframe
        assert not washover_df.query("time >= '2022-01-01 01:00:00'").equals(
            out_df.query("time >= '2022-01-01 01:00:00'")
        )
