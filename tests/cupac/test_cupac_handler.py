from datetime import date

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from cluster_experiments.cupac import CupacHandler, TargetAggregation
from tests.utils import generate_random_data

N = 1_000


@pytest.fixture
def clusters():
    return [f"Cluster {i}" for i in range(100)]


@pytest.fixture
def dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]


@pytest.fixture
def experiment_dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]


@pytest.fixture
def df(clusters, dates):
    return generate_random_data(clusters, dates, N)


@pytest.fixture
def df_feats(clusters, dates):
    df = generate_random_data(clusters, dates, N)
    df["x1"] = np.random.normal(0, 1, N)
    df["x2"] = np.random.normal(0, 1, N)
    return df


@pytest.fixture
def cupac_handler_base():
    return CupacHandler(
        cupac_model=TargetAggregation(
            agg_col="user",
        ),
        target_col="target",
        # features_cupac_model=["user"],
    )


@pytest.fixture
def cupac_handler_model():
    return CupacHandler(
        cupac_model=HistGradientBoostingRegressor(max_iter=3),
        target_col="target",
        features_cupac_model=["x1", "x2"],
    )


@pytest.fixture
def missing_cupac():
    return CupacHandler(
        None,
    )


@pytest.mark.parametrize(
    "cupac_handler",
    [
        "cupac_handler_base",
        "cupac_handler_model",
    ],
)
def test_add_covariates(cupac_handler, df_feats, request):
    cupac_handler = request.getfixturevalue(cupac_handler)
    df = cupac_handler.add_covariates(df_feats, df_feats.head(10))
    assert df["estimate_target"].isna().sum() == 0
    assert (df["estimate_target"] <= df["target"].max()).all()
    assert (df["estimate_target"] >= df["target"].min()).all()


def test_no_target(missing_cupac, df_feats):
    """Checks that no target is added when the there is no cupac model"""
    df = missing_cupac.add_covariates(df_feats)
    assert "estimate_target" not in df.columns


def test_no_pre_experiment(cupac_handler_base, df_feats):
    """Checks that if there is a cupac model, pre_experiment_df should be provided"""
    with pytest.raises(ValueError, match="pre_experiment_df should be provided"):
        cupac_handler_base.add_covariates(df_feats)


def test_no_cupac(missing_cupac, df_feats):
    """Checks that if there is no cupac model, pre_experiment_df should not be provided"""
    with pytest.raises(ValueError, match="remove pre_experiment_df argument"):
        missing_cupac.add_covariates(df_feats, df_feats.head(10))
