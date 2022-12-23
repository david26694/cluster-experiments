import pandas as pd
import pytest

from tests.splitter.conftest import (
    balanced_splitter_parametrize,
    stratified_splitter_parametrize,
    switchback_splitter_parametrize,
)

add_cluster_cols_parametrize = pytest.mark.parametrize(
    "add_cluster_cols", [True, False]
)


@switchback_splitter_parametrize
def test_switchback_splitter(splitter, date_df, request):
    switchback_splitter = request.getfixturevalue(splitter)

    treatment_assignment = switchback_splitter.assign_treatment_df(date_df)
    assert "time" in switchback_splitter.cluster_cols

    # Only 1 treatment per date
    assert (treatment_assignment.groupby("time")["treatment"].nunique() == 1).all()


@switchback_splitter_parametrize
def test_clustered_switchback_splitter(splitter, biweekly_df, request):
    switchback_splitter = request.getfixturevalue(splitter)
    biweekly_df_long = pd.concat([biweekly_df for _ in range(3)])
    switchback_splitter.cluster_cols = ["cluster"]
    treatment_assignment = switchback_splitter.assign_treatment_df(biweekly_df_long)
    assert "time" in switchback_splitter.cluster_cols

    # Only 1 treatment per cluster
    assert (
        treatment_assignment.groupby(["cluster", "time"])["treatment"].nunique() == 1
    ).all()


@balanced_splitter_parametrize
@add_cluster_cols_parametrize
def test_clustered_switchback_splitter_balance(
    splitter, add_cluster_cols, biweekly_df, request
):
    balanced_splitter = request.getfixturevalue(splitter)

    if add_cluster_cols:
        balanced_splitter.cluster_cols = ["cluster"]
    treatment_assignment = balanced_splitter.assign_treatment_df(biweekly_df)
    assert "time" in balanced_splitter.cluster_cols
    # Assert that the treatment assignment is balanced
    assert (treatment_assignment.treatment.value_counts() == 70).all()


@stratified_splitter_parametrize
@add_cluster_cols_parametrize
def test_stratified_splitter(splitter, add_cluster_cols, biweekly_df, request):
    stratified_switchback_splitter = request.getfixturevalue(splitter)

    if add_cluster_cols:
        stratified_switchback_splitter.cluster_cols = ["cluster"]
        stratified_switchback_splitter.strata_cols += ["cluster"]

    treatment_assignment = stratified_switchback_splitter.assign_treatment_df(
        biweekly_df
    )
    assert "time" in stratified_switchback_splitter.cluster_cols
    assert (treatment_assignment.treatment.value_counts() == 70).all()
    # Per cluster, there are 2 treatments. Per day of week too
    for col in ["cluster", "day_of_week"]:
        assert (treatment_assignment.groupby([col])["treatment"].nunique() == 2).all()

    # Check stratification. Count day_of_week and treatment, we should always
    # have the same number of observations. Same for cluster
    for col in ["cluster", "day_of_week"]:
        assert treatment_assignment.groupby([col, "treatment"]).size().nunique() == 1
