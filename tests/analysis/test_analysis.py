import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    ConfidenceInterval,
    DeltaMethodAnalysis,
    ExperimentAnalysis,
    GeeExperimentAnalysis,
    InferenceResults,
    MLMExperimentAnalysis,
    OLSAnalysis,
    PairedTTestClusteredAnalysis,
    TTestClusteredAnalysis,
)
from tests.utils import (
    generate_clustered_data,
    generate_random_data,
    generate_ratio_metric_data,
)


@pytest.fixture
def analysis_df_diff():
    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "target": [0] * 16,
        }
    )
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analysis_df_full.loc[analysis_df_full["treatment"] == "B", "target"] = 0.1
    return analysis_df_full


@pytest.fixture
def analysis_df_diff_realistic():
    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "target": [1] * 16,
        }
    )
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analysis_df_full.loc[analysis_df_full["treatment"] == "B", "target"] = 1.01
    np.random.seed(2024)
    analysis_df_full["target"] = analysis_df_full["target"] + np.random.normal(
        0, 0.1, analysis_df_full.shape[0]
    )
    return analysis_df_full


def test_cluster_column(analysis_df):
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert (analyser._get_cluster_column(analysis_df) == "Cluster 12022-01-01").all()


def test_binary_treatment(analysis_df):
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert (
        analyser._create_binary_treatment(analysis_df)["treatment"]
        == pd.Series([0, 1, 1, 0])
    ).all()


def test_get_pvalue(analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert analyser.get_pvalue(analysis_df_full) >= 0


def test_pvalue_based_on_hypothesis():
    analyser = GeeExperimentAnalysis(cluster_cols=["cluster"])

    class MockStatsModelResult:
        def __init__(self, params, pvalues):
            self.params = params
            self.pvalues = pvalues

    mock_result = MockStatsModelResult(
        params={"treatment": -0.5},  # Example treatment effect
        pvalues={"treatment": 0.04},  # Example p-value
    )
    analyser.hypothesis = "less"

    result = analyser.pvalue_based_on_hypothesis(mock_result)

    # Instance of the class containing the method
    # Set hypothesis as needed

    # Assert the expected outcome
    expected_pvalue = 0.02  # Expected p-value based on your method's logic
    assert result == pytest.approx(expected_pvalue), "P-value calculation is incorrect"


def test_mlm_analysis(analysis_df_diff):
    analyser = MLMExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    # Changing just one observation so we have a p value
    analysis_df_diff.loc[1, "target"] = 0.00001

    p_value = analyser.get_pvalue(analysis_df_diff)
    point_estimate = analyser.get_point_estimate(analysis_df_diff)
    assert np.isclose(p_value, 0, atol=1e-5)
    assert np.isclose(point_estimate, 0.1, atol=1e-5)


def test_ttest(analysis_df_diff):

    analyser = TTestClusteredAnalysis(cluster_cols=["cluster"])

    assert 0.05 >= analyser.get_pvalue(analysis_df_diff) >= 0


def test_paired_ttest():
    "This test make sure that pivot table works as expected (inside get_pvalue) and that paired t test returns a possible value"

    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"] * 4,
            "treatment": ["A", "B"] * 8,
            "target": [0.01] * 16,
        }
    )
    # Changing just one observation so we have a p value
    analysis_df.loc[1, "target"] = 0.001

    analyser = PairedTTestClusteredAnalysis(
        cluster_cols=["cluster", "date"], strata_cols=["cluster"]
    )

    assert 1 >= analyser.get_pvalue(analysis_df) >= 0, "p value is null or inf"


def test_paired_ttest_preprocessing():

    analyser = PairedTTestClusteredAnalysis(
        cluster_cols=["country_code", "city_code"], strata_cols=["country_code"]
    )
    df = generate_clustered_data()

    df_pivot = analyser._preprocessing(df=df)

    assert df_pivot.isna().sum().sum() == 0, "Unexpected nas in pivot"
    assert (df_pivot.index == ["ES", "IT", "PL", "RO"]).all(), "wrong index"
    assert (df_pivot.columns == ["A", "B"]).all(), "wrong columns"
    assert (
        df_pivot.values == [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.1, 0.01]]
    ).all(), "wrong values"

    assert df_pivot.shape == (
        df["country_code"].nunique(),
        2,
    ), "different shape than expected"


def test_ttest_random_data():
    N = 1000
    analysis_df = generate_random_data(
        clusters=[f"c_{i}" for i in range(100)], dates=["2021-01-01"], N=N
    ).assign(treatment=np.random.choice(["A", "B"], size=N))
    analyser = TTestClusteredAnalysis(cluster_cols=["cluster", "date"])

    assert analyser.get_pvalue(analysis_df) >= 0


def test_point_estimate_raises():
    class DummyAnalysis(ExperimentAnalysis):
        def __init__(self):
            pass

        def analysis_pvalue(self, df):
            return 0.05

    analyser = DummyAnalysis()
    with pytest.raises(NotImplementedError):
        analyser.analysis_point_estimate(df=pd.DataFrame())


def test_confidence_interval():
    conf_int = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
    assert conf_int.lower == 0.1
    assert conf_int.upper == 0.5
    assert conf_int.alpha == 0.05


def test_inference_results():
    conf_int = ConfidenceInterval(lower=0.1, upper=0.5, alpha=0.05)
    results = InferenceResults(ate=0.2, p_value=0.03, std_error=0.1, conf_int=conf_int)
    assert results.ate == 0.2
    assert results.p_value == 0.03
    assert results.std_error == 0.1
    assert results.conf_int == conf_int


@pytest.mark.parametrize(
    "experiment_analysis",
    [
        ClusteredOLSAnalysis(
            cluster_cols=["cluster"], target_col="target", treatment_col="treatment"
        ),
        GeeExperimentAnalysis(
            cluster_cols=["cluster"], target_col="target", treatment_col="treatment"
        ),
        OLSAnalysis(target_col="target", treatment_col="treatment"),
    ],
)  # Add other child classes as necessary
def test_get_confidence_interval(experiment_analysis, analysis_df_diff_realistic):

    # Check if the get_confidence_interval method works
    alpha = 0.05
    conf_int = experiment_analysis.get_confidence_interval(
        analysis_df_diff_realistic, alpha=alpha
    )

    assert isinstance(conf_int, ConfidenceInterval)
    assert conf_int.alpha == alpha
    assert conf_int.lower < conf_int.upper  # Simple sanity check


@pytest.mark.parametrize(
    "experiment_analysis",
    [
        ClusteredOLSAnalysis(
            cluster_cols=["cluster"], target_col="target", treatment_col="treatment"
        ),
        GeeExperimentAnalysis(
            cluster_cols=["cluster"], target_col="target", treatment_col="treatment"
        ),
        OLSAnalysis(target_col="target", treatment_col="treatment"),
    ],
)  # Add other child classes as necessary
def test_get_inference_results(experiment_analysis, analysis_df_diff):
    # Check if the get_inference_results method works
    alpha = 0.05
    results = experiment_analysis.get_inference_results(analysis_df_diff, alpha=alpha)

    assert isinstance(results, InferenceResults)
    assert results.conf_int.alpha == alpha
    assert results.ate > 0
    assert 0 <= results.p_value < 1
    assert results.std_error > 0


@pytest.fixture
def analysis_df_ratio():
    analysis_df = pd.DataFrame(
        {
            "cluster": [1, 2, 3, 1, 4, 5, 5, 6] * 2,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "target": [1, 0, 1, 0, 1, 1, 1, 1] * 2,
            "scale": [1, 1, 1, 1, 1, 1, 1, 1] * 2,
        }
    )
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    return analysis_df_full


def test_delta_analysis(analysis_ratio_df, experiment_dates):
    analyser = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")
    experiment_start = min(experiment_dates)

    df_experiment = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    assert 0.05 >= analyser.get_pvalue(df_experiment) >= 0


def test_aa_delta_analysis(dates):

    analyser = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")

    p_values = []
    for _ in range(100):
        data = generate_ratio_metric_data(dates, N=100_000, treatment_effect=0)
        p_values.append(analyser.get_pvalue(data))

    positive_rate = sum(p < 0.05 for p in p_values) / len(p_values)

    assert positive_rate == pytest.approx(
        0.05, abs=0.01
    ), "P-value A/A calculation is incorrect"


def test_cuped_delta_analysis(analysis_ratio_df, experiment_dates):
    experiment_start_date = min(experiment_dates)
    analyser = DeltaMethodAnalysis(
        ratio_covariates=[("pre_target", "pre_scale")],
        scale_col="scale",
    )
    # TODO: Move to preprocessing in class?
    pre_df = (
        analysis_ratio_df.query(f" date < '{experiment_start_date}'")
        .groupby(["user"], as_index=False)
        .agg(pre_target=("target", "sum"), pre_scale=("scale", "sum"))
    )
    df = (
        analysis_ratio_df.query(f"date >= '{experiment_start_date}'")
        .groupby(["user", "treatment"], as_index=False)
        .agg(target=("target", "sum"), scale=("scale", "sum"))
    )

    df = df.merge(pre_df, on="user", how="left")
    df["pre_target"] = df["pre_target"].fillna(df["pre_target"].mean())
    df["pre_scale"] = df["pre_scale"].fillna(df["pre_scale"].mean())

    assert 0.05 >= analyser.get_pvalue(df) >= 0


def test_aa_cuped_delta_analysis(dates, experiment_dates):
    experiment_start_date = min(experiment_dates)

    p_values = []
    for _ in range(1000):
        analyser = DeltaMethodAnalysis(
            scale_col="scale",
            ratio_covariates=[("pre_target", "pre_scale")],
        )
        data = generate_ratio_metric_data(dates, N=100_000, treatment_effect=0)
        # TODO: Move to preprocessing in class?
        pre_df = (
            data.query(f" date < '{experiment_start_date}'")
            .groupby(["user"], as_index=False)
            .agg(pre_target=("target", "sum"), pre_scale=("scale", "sum"))
        )
        df = (
            data.query(f"date >= '{experiment_start_date}'")
            .groupby(["user", "treatment"], as_index=False)
            .agg(target=("target", "sum"), scale=("scale", "sum"))
        )

        df = df.merge(pre_df, on="user", how="left")
        df["pre_target"] = df["pre_target"].fillna(df["pre_target"].mean())
        df["pre_scale"] = df["pre_scale"].fillna(df["pre_scale"].mean())

        p_values.append(analyser.get_pvalue(df))

    positive_rate = sum(p < 0.05 for p in p_values) / len(p_values)

    assert positive_rate == pytest.approx(
        0.05, abs=0.01
    ), "P-value A/A calculation is incorrect"


def test_stats_delta_vs_ols(analysis_ratio_df, experiment_dates):
    experiment_start_date = min(experiment_dates)

    analyser_ols = ClusteredOLSAnalysis(cluster_cols=["user"])
    analyser_delta = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")
    df = analysis_ratio_df.query(f"date >= '{experiment_start_date}'")

    point_estimate_ols = analyser_ols.get_point_estimate(df)
    point_estimate_delta = analyser_delta.get_point_estimate(df)

    SE_ols = analyser_ols.get_standard_error(df)
    SE_delta = analyser_delta.get_standard_error(df)

    assert point_estimate_delta == pytest.approx(
        point_estimate_ols, rel=1e-2
    ), "Point estimate is not consistent with Clustered OLS"
    assert SE_delta == pytest.approx(
        SE_ols, rel=1e-2
    ), "Standard error is not consistent with Clustered OLS"


def test_stats_cuped_delta_vs_ols(analysis_ratio_df, experiment_dates):

    from statsmodels.formula.api import ols

    experiment_start_date = min(experiment_dates)

    # TODO: Move to preprocessing in class?
    pre_df = (
        analysis_ratio_df.query(f" date < '{experiment_start_date}'")
        .groupby(["user"], as_index=False)
        .agg(pre_target=("target", "sum"), pre_scale=("scale", "sum"))
    )
    df = (
        analysis_ratio_df.query(f"date >= '{experiment_start_date}'")
        .groupby(["user", "treatment"], as_index=False)
        .agg(target=("target", "sum"), scale=("scale", "sum"))
    )

    df = df.merge(pre_df, on="user", how="left")
    df["pre_target"] = df["pre_target"].fillna(df["pre_target"].mean())
    df["pre_scale"] = df["pre_scale"].fillna(df["pre_scale"].mean())

    analyser_delta = DeltaMethodAnalysis(
        ratio_covariates=[("pre_target", "pre_scale")],
        scale_col="scale",
    )

    effect_delta = analyser_delta.get_point_estimate(df)

    analyser_delta = DeltaMethodAnalysis(
        ratio_covariates=[("pre_target", "pre_scale")],
        scale_col="scale",
    )

    SE_delta = analyser_delta.get_standard_error(df)

    df["treatment"] = df["treatment"].map({"A": 0, "B": 1})

    results = ols(
        "Y ~ X + d",
        {
            "Y": df.target / df.scale,
            "X": (df.pre_target / df.pre_scale - (df.pre_target / df.pre_scale).mean()),
            "d": df.treatment,
        },
    ).fit(cov_type="cluster", cov_kwds={"groups": df["user"], "use_correction": False})
    effect_ols = results.params["d"]
    SE_ols = results.bse["d"]

    assert effect_delta == pytest.approx(
        effect_ols, rel=1e-2
    ), "CUPED Point estimate is not consistent with OLS"
    assert SE_delta == pytest.approx(
        SE_ols, rel=1e-2
    ), "CUPED Standard error is not consistent with OLS"
