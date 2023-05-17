import logging

from cluster_experiments.power_config import PowerConfig


def test_config_warning_superfluous_param_switch_frequency(caplog):
    msg = "switch_frequency = 1H has no effect with splitter = non_clustered. Overriding switch_frequency to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            switch_frequency="1H",
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_washover_time_delta(caplog):
    msg = "washover_time_delta = 30 has no effect with splitter = non_clustered. Overriding washover_time_delta to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            washover_time_delta=30,
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_washover(caplog):
    msg = "washover = constant_washover has no effect with splitter = non_clustered. Overriding washover to ."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            washover="constant_washover",
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_time_col(caplog):
    msg = "time_col = datetime has no effect with splitter = non_clustered. Overriding time_col to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            time_col="datetime",
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_perturbator(caplog):
    msg = "scale = 0.5 has no effect with perturbator = uniform. Overriding scale to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            scale=0.5,
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_strata_cols(caplog):
    msg = "strata_cols = ['group'] has no effect with splitter = non_clustered. Overriding strata_cols to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            strata_cols=["group"],
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_agg_col(caplog):
    msg = "agg_col = agg_col has no effect with cupac_model = . Overriding agg_col to ."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            agg_col="agg_col",
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_smoothing_factor(caplog):
    msg = "smoothing_factor = 0.5 has no effect with cupac_model = . Overriding smoothing_factor to 20."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            smoothing_factor=0.5,
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_features_cupac_model(caplog):
    msg = "features_cupac_model = ['feature1'] has no effect with cupac_model = . Overriding features_cupac_model to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ols_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            features_cupac_model=["feature1"],
        )
    assert msg in caplog.text


def test_config_warning_superfluous_param_covariates(caplog):
    msg = "covariates = ['covariate1'] has no effect with analysis = ttest_clustered. Overriding covariates to None."
    with caplog.at_level(logging.WARNING):
        PowerConfig(
            cluster_cols=["cluster", "date"],
            analysis="ttest_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
            covariates=["covariate1"],
        )
    assert msg in caplog.text
