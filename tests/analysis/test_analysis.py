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
)


@pytest.fixture
def analysis_df_diff():
    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "scale": [1] * 16,
            "target": [0] * 16,
        }
    )
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    np.random.seed(2024)
    analysis_df_full.loc[analysis_df_full["treatment"] == "B", "target"] = (
        0.1 + np.random.normal(0, 0.00001, (analysis_df_full["treatment"] == "B").sum())
    )
    return analysis_df_full


@pytest.fixture
def analysis_df_diff_realistic():
    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "scale": [1] * 16,
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
        DeltaMethodAnalysis(cluster_cols=["cluster"], scale_col="scale"),
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
        DeltaMethodAnalysis(cluster_cols=["cluster"], scale_col="scale"),
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
