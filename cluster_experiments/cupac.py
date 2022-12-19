from typing import List, Optional, Tuple

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class EmptyRegressor(BaseEstimator, RegressorMixin):
    """
    Empty regressor class. It does not do anything, used to glue the code of other estimators and PowerAnalysis

    Each Regressor should have:
    - fit method: Uses pre experiment data to fit some kind of model to be used as a covariate and reduce variance.
    - predict method: Uses the fitted model to add the covariate on the experiment data.

    It can add aggregates of the target in older data as a covariate, or a model (cupac) to predict the target.
    """

    @classmethod
    def from_config(cls, config):
        return cls()


class TargetAggregation(BaseEstimator, RegressorMixin):
    """
    Adds average of target using pre-experiment data

    Args:
        agg_col: Column to group by to aggregate target
        target_col: Column to aggregate
        smoothing_factor: Smoothing factor for the smoothed mean
    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.cupac import TargetAggregation

    df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"], "target_col": [1, 2, 3, 4, 5, 6]})
    new_df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"]})
    target_agg = TargetAggregation("agg_col", "target_col")
    target_agg.fit(df.drop(columns="target_col"), df["target_col"])
    df_with_target_agg = target_agg.predict(new_df)
    print(df_with_target_agg)
    ```
    """

    def __init__(
        self,
        agg_col: str,
        target_col: str = "target",
        smoothing_factor: int = 20,
    ):
        self.agg_col = agg_col
        self.target_col = target_col
        self.smoothing_factor = smoothing_factor
        self.is_empty = False
        self.mean_target_col = f"{self.target_col}_mean"
        self.smooth_mean_target_col = f"{self.target_col}_smooth_mean"
        self.pre_experiment_agg_df = pd.DataFrame()

    def _get_pre_experiment_mean(self, pre_experiment_df: pd.DataFrame) -> float:
        return pre_experiment_df[self.target_col].mean()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetAggregation":
        """Fits "target encoder" model to pre-experiment data"""
        pre_experiment_df = X.copy()
        pre_experiment_df[self.target_col] = y

        self.pre_experiment_mean = self._get_pre_experiment_mean(pre_experiment_df)
        self.pre_experiment_agg_df = (
            pre_experiment_df.assign(count=1)
            .groupby(self.agg_col, as_index=False)
            .agg({self.target_col: "sum", "count": "sum"})
            .assign(
                **{
                    self.mean_target_col: lambda x: x[self.target_col] / x["count"],
                    self.smooth_mean_target_col: lambda x: (
                        x[self.target_col]
                        + self.smoothing_factor * self.pre_experiment_mean
                    )
                    / (x["count"] + self.smoothing_factor),
                }
            )
            .drop(columns=["count", self.target_col])
        )
        return self

    def predict(self, X: pd.DataFrame) -> ArrayLike:
        """Adds average target of pre-experiment data to experiment data"""
        return (
            X.merge(self.pre_experiment_agg_df, how="left", on=self.agg_col)[
                self.smooth_mean_target_col
            ]
            .fillna(self.pre_experiment_mean)
            .values
        )

    @classmethod
    def from_config(cls, config):
        """Creates TargetAggregation from PowerConfig"""
        return cls(
            agg_col=config.agg_col,
            target_col=config.target_col,
            smoothing_factor=config.smoothing_factor,
        )


class CupacHandler:
    """
    CupacHandler class. It handles operations related to the cupac model.

    Its main goal is to call the add_covariates method, where it will add the ouptut from the cupac model,
    and this should be used as covariates in the regression method for the hypothesis test.
    """

    def __init__(
        self,
        cupac_model: Optional[BaseEstimator] = None,
        target_col: str = "target",
        features_cupac_model: Optional[List[str]] = None,
    ):
        self.cupac_model: BaseEstimator = cupac_model or EmptyRegressor()
        self.target_col = target_col
        self.cupac_outcome_name = f"estimate_{target_col}"
        self.features_cupac_model: List[str] = features_cupac_model or []
        self.is_cupac = not isinstance(self.cupac_model, EmptyRegressor)

    def _prep_data_cupac(
        self, df: pd.DataFrame, pre_experiment_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepares data for training and prediction"""
        df = df.copy()
        pre_experiment_df = pre_experiment_df.copy()
        df_predict = df.drop(columns=[self.target_col])
        # Split data into X and y
        pre_experiment_x = pre_experiment_df.drop(columns=[self.target_col])
        pre_experiment_y = pre_experiment_df[self.target_col]

        # Keep only cupac features
        if self.features_cupac_model:
            pre_experiment_x = pre_experiment_x[self.features_cupac_model]
            df_predict = df_predict[self.features_cupac_model]

        return df_predict, pre_experiment_x, pre_experiment_y

    def add_covariates(
        self, df: pd.DataFrame, pre_experiment_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Train model to predict outcome variable (based on pre-experiment data)
        and  add the prediction to the experiment dataframe. Only do this if
        we use cupac
        Args:
            pre_experiment_df: Dataframe with pre-experiment data.
            df: Dataframe with outcome and treatment variables.
        """
        self.check_cupac_inputs(pre_experiment_df)

        # Early return if no need to add covariates
        if not self.need_covariates(pre_experiment_df):
            return df

        df = df.copy()
        pre_experiment_df = pre_experiment_df.copy()
        df_predict, pre_experiment_x, pre_experiment_y = self._prep_data_cupac(
            df=df, pre_experiment_df=pre_experiment_df
        )

        # Fit model
        self.cupac_model.fit(pre_experiment_x, pre_experiment_y)

        # Predict
        if isinstance(self.cupac_model, RegressorMixin):
            estimated_target = self.cupac_model.predict(df_predict)
        elif isinstance(self.cupac_model, ClassifierMixin):
            estimated_target = self.cupac_model.predict_proba(df_predict)[:, 1]
        else:
            raise ValueError(
                "cupac_model should be an instance of RegressorMixin or ClassifierMixin"
            )

        # Add cupac outcome name to df
        df[self.cupac_outcome_name] = estimated_target
        return df

    def need_covariates(self, pre_experiment_df: Optional[pd.DataFrame] = None) -> bool:
        return pre_experiment_df is not None and self.is_cupac

    def check_cupac_inputs(self, pre_experiment_df: Optional[pd.DataFrame] = None):
        if self.is_cupac and pre_experiment_df is None:
            raise ValueError("If cupac is used, pre_experiment_df should be provided.")
