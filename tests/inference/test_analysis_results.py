from dataclasses import asdict

import pandas as pd
import pytest

from cluster_experiments.inference.analysis_results import AnalysisPlanResults


def test_analysis_plan_results_initialization():
    """Test that AnalysisPlanResults initializes with empty lists by default."""
    results = AnalysisPlanResults()
    assert results.metric_alias == []
    assert results.control_variant_name == []
    assert results.treatment_variant_name == []
    assert results.control_variant_mean == []
    assert results.treatment_variant_mean == []
    assert results.analysis_type == []
    assert results.ate == []
    assert results.ate_ci_lower == []
    assert results.ate_ci_upper == []
    assert results.p_value == []
    assert results.std_error == []
    assert results.dimension_name == []
    assert results.dimension_value == []
    assert results.alpha == []


def test_analysis_plan_results_custom_initialization():
    """Test that AnalysisPlanResults initializes with custom data."""
    results = AnalysisPlanResults(
        metric_alias=["metric1"],
        control_variant_name=["Control"],
        treatment_variant_name=["Treatment"],
        control_variant_mean=[0.5],
        treatment_variant_mean=[0.6],
        analysis_type=["AB Test"],
        ate=[0.1],
        ate_ci_lower=[0.05],
        ate_ci_upper=[0.15],
        p_value=[0.04],
        std_error=[0.02],
        dimension_name=["Country"],
        dimension_value=["US"],
        alpha=[0.05],
    )
    assert results.metric_alias == ["metric1"]
    assert results.control_variant_name == ["Control"]
    assert results.treatment_variant_name == ["Treatment"]
    assert results.control_variant_mean == [0.5]
    assert results.treatment_variant_mean == [0.6]
    assert results.analysis_type == ["AB Test"]
    assert results.ate == [0.1]
    assert results.ate_ci_lower == [0.05]
    assert results.ate_ci_upper == [0.15]
    assert results.p_value == [0.04]
    assert results.std_error == [0.02]
    assert results.dimension_name == ["Country"]
    assert results.dimension_value == ["US"]
    assert results.alpha == [0.05]


def test_analysis_plan_results_addition():
    """Test that two AnalysisPlanResults instances can be added together."""
    results1 = AnalysisPlanResults(
        metric_alias=["metric1"],
        control_variant_name=["Control"],
        treatment_variant_name=["Treatment"],
        control_variant_mean=[0.5],
        treatment_variant_mean=[0.6],
        analysis_type=["AB Test"],
        ate=[0.1],
        ate_ci_lower=[0.05],
        ate_ci_upper=[0.15],
        p_value=[0.04],
        std_error=[0.02],
        dimension_name=["Country"],
        dimension_value=["US"],
        alpha=[0.05],
    )
    results2 = AnalysisPlanResults(
        metric_alias=["metric2"],
        control_variant_name=["Control"],
        treatment_variant_name=["Treatment"],
        control_variant_mean=[0.55],
        treatment_variant_mean=[0.65],
        analysis_type=["AB Test"],
        ate=[0.1],
        ate_ci_lower=[0.05],
        ate_ci_upper=[0.15],
        p_value=[0.03],
        std_error=[0.01],
        dimension_name=["Country"],
        dimension_value=["CA"],
        alpha=[0.05],
    )
    combined_results = results1 + results2

    assert combined_results.metric_alias == ["metric1", "metric2"]
    assert combined_results.control_variant_name == ["Control", "Control"]
    assert combined_results.treatment_variant_name == ["Treatment", "Treatment"]
    assert combined_results.control_variant_mean == [0.5, 0.55]
    assert combined_results.treatment_variant_mean == [0.6, 0.65]
    assert combined_results.analysis_type == ["AB Test", "AB Test"]
    assert combined_results.ate == [0.1, 0.1]
    assert combined_results.ate_ci_lower == [0.05, 0.05]
    assert combined_results.ate_ci_upper == [0.15, 0.15]
    assert combined_results.p_value == [0.04, 0.03]
    assert combined_results.std_error == [0.02, 0.01]
    assert combined_results.dimension_name == ["Country", "Country"]
    assert combined_results.dimension_value == ["US", "CA"]
    assert combined_results.alpha == [0.05, 0.05]


def test_analysis_plan_results_addition_type_error():
    """Test that adding a non-AnalysisPlanResults object raises a TypeError."""
    results = AnalysisPlanResults(metric_alias=["metric1"])
    with pytest.raises(TypeError):
        results + "not_an_analysis_plan_results"  # Should raise TypeError


def test_analysis_plan_results_to_dataframe():
    """Test that AnalysisPlanResults converts to a DataFrame correctly."""
    results = AnalysisPlanResults(
        metric_alias=["metric1"],
        control_variant_name=["Control"],
        treatment_variant_name=["Treatment"],
        control_variant_mean=[0.5],
        treatment_variant_mean=[0.6],
        analysis_type=["AB Test"],
        ate=[0.1],
        ate_ci_lower=[0.05],
        ate_ci_upper=[0.15],
        p_value=[0.04],
        std_error=[0.02],
        dimension_name=["Country"],
        dimension_value=["US"],
        alpha=[0.05],
    )
    df = results.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1  # Only one entry
    assert set(df.columns) == set(asdict(results).keys())  # Columns match attributes
    assert df["metric_alias"].iloc[0] == "metric1"
    assert df["control_variant_name"].iloc[0] == "Control"
    assert df["treatment_variant_name"].iloc[0] == "Treatment"
    assert df["control_variant_mean"].iloc[0] == 0.5
    assert df["treatment_variant_mean"].iloc[0] == 0.6
    assert df["ate"].iloc[0] == 0.1
    assert df["ate_ci_lower"].iloc[0] == 0.05
    assert df["ate_ci_upper"].iloc[0] == 0.15
    assert df["p_value"].iloc[0] == 0.04
    assert df["std_error"].iloc[0] == 0.02
    assert df["dimension_name"].iloc[0] == "Country"
    assert df["dimension_value"].iloc[0] == "US"
    assert df["alpha"].iloc[0] == 0.05
