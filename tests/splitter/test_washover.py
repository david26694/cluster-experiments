from datetime import timedelta

import pytest

from cluster_experiments.washover import ConstantWashover


@pytest.mark.parametrize("minutes, n_rows", [(30, 2), (10, 4), (15, 3)])
def test_constant_washover_base(minutes, n_rows, washover_base_df):

    out_df = ConstantWashover(washover_time_delta=timedelta(minutes=minutes)).washover(
        df=washover_base_df,
        time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
    )

    assert len(out_df) == n_rows
    assert (out_df["og___time"].dt.minute > minutes).all()


@pytest.mark.parametrize(
    "minutes, n_rows",
    [
        (30, 4),
    ],
)
def test_constant_washover_no_switch(minutes, n_rows, washover_df_no_switch):

    out_df = ConstantWashover(washover_time_delta=timedelta(minutes=minutes)).washover(
        df=washover_df_no_switch,
        time_col="time",
        cluster_cols=["city", "time"],
        treatment_col="treatment",
    )
    assert len(out_df) == n_rows
    # Check that, after 2022-01-01 02:00:00, we keep all the rows of the original
    # dataframe
    assert washover_df_no_switch.query("time >= '2022-01-01 02:00:00'").equals(
        out_df.query("time >= '2022-01-01 02:00:00'")
    )

    # Check that, after 2022-01-01 01:00:00, we don't have the same rows as the
    # original dataframe
    assert not washover_df_no_switch.query("time >= '2022-01-01 01:00:00'").equals(
        out_df.query("time >= '2022-01-01 01:00:00'")
    )
