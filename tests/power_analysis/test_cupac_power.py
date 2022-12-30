from sklearn.ensemble import HistGradientBoostingRegressor


def test_power_analyis_aggregate(df, experiment_dates, cupac_power_analysis):
    df_analysis = df.query(f"date.isin({experiment_dates})")
    df_pre = df.query(f"~date.isin({experiment_dates})")
    power = cupac_power_analysis.power_analysis(df_analysis, df_pre)
    assert power >= 0
    assert power <= 1


def test_add_covariates(df, experiment_dates, cupac_power_analysis):
    df_analysis = df.query(f"date.isin({experiment_dates})")
    df_pre = df.query(f"~date.isin({experiment_dates})")
    estimated_target = cupac_power_analysis.cupac_handler.add_covariates(
        df_analysis, df_pre
    )["estimate_target"]
    assert estimated_target.isnull().sum() == 0
    assert (estimated_target <= df_pre["target"].max()).all()
    assert (estimated_target >= df_pre["target"].min()).all()
    assert "estimate_target" in cupac_power_analysis.analysis.covariates


def test_prep_data(df_feats, experiment_dates, cupac_power_analysis):
    df = df_feats.copy()
    df_analysis = df.query(f"date.isin({experiment_dates})")
    df_pre = df.query(f"~date.isin({experiment_dates})")
    cupac_power_analysis.cupac_handler.features_cupac_model = ["x1", "x2"]
    (
        df_predict,
        pre_experiment_x,
        pre_experiment_y,
    ) = cupac_power_analysis.cupac_handler._prep_data_cupac(df_analysis, df_pre)
    assert list(df_predict.columns) == ["x1", "x2"]
    assert list(pre_experiment_x.columns) == ["x1", "x2"]
    assert (df_predict["x1"] == df_analysis["x1"]).all()
    assert (pre_experiment_x["x1"] == df_pre["x1"]).all()
    assert (pre_experiment_y == df_pre["target"]).all()


def test_cupac_gbm(df_feats, experiment_dates, cupac_power_analysis):
    df = df_feats.copy()
    df_analysis = df.query(f"date.isin({experiment_dates})")
    df_pre = df.query(f"~date.isin({experiment_dates})")
    cupac_power_analysis.features_cupac_model = ["x1", "x2"]
    cupac_power_analysis.cupac_model = HistGradientBoostingRegressor()
    power = cupac_power_analysis.power_analysis(df_analysis, df_pre)
    assert power >= 0
    assert power <= 1
