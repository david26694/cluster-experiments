from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils.validation import NotFittedError, check_is_fitted


@runtime_checkable
class MLHandler(Protocol):
    @property
    def cupac_outcome_name(self) -> str: ...

    def add_covariates(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame: ...


class NoOpHandler:
    @property
    def cupac_outcome_name(self) -> str:
        return ""

    def add_covariates(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return df


class EmptyRegressor(BaseEstimator):
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


class TargetAggregation(BaseEstimator):
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
        scale_col: Optional[str] = None,
        features_cupac_model: Optional[List[str]] = None,
        cache_fit: bool = True,
    ):
        self.cupac_model: BaseEstimator = cupac_model or EmptyRegressor()
        self.target_col = target_col
        # TODO: implement CUPAC with both target_col and scale_col,
        # right now it only supports target_col for delta method
        self.scale_col = scale_col
        self.cupac_outcome_name = f"estimate_{target_col}"
        self.features_cupac_model: List[str] = features_cupac_model or []
        self.is_cupac = not isinstance(self.cupac_model, EmptyRegressor)
        self.cache_fit = cache_fit

        self.check_cupac_config()

    def get_pre_experiment_y(self, pre_experiment_df: pd.DataFrame) -> pd.Series:
        """Returns the pre-experiment target variable, scaled if scale_col is provided."""
        if self.scale_col is not None:
            return (
                pre_experiment_df[self.target_col] / pre_experiment_df[self.scale_col]
            )
        return pre_experiment_df[self.target_col]

    def _prep_data_cupac(
        self, df: pd.DataFrame, pre_experiment_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepares data for training and prediction"""
        df = df.copy()
        pre_experiment_df = pre_experiment_df.copy()
        df_predict = df.drop(columns=[self.target_col])
        # Split data into X and y
        pre_experiment_x = pre_experiment_df.drop(columns=[self.target_col])
        pre_experiment_y = self.get_pre_experiment_y(pre_experiment_df)

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

        # Fit model if it has not been fitted before
        self._fit_cupac_model(pre_experiment_x, pre_experiment_y)

        # Predict
        estimated_target = self._predict_cupac_model(df_predict)

        # Add cupac outcome name to df
        df[self.cupac_outcome_name] = estimated_target
        return df

    def _fit_cupac_model(
        self, pre_experiment_x: pd.DataFrame, pre_experiment_y: pd.Series
    ):
        """Fits the cupac model.
        Caches the fitted model in the object, so we only fit it once.
        We can disable this by setting cache_fit to False.
        """
        if not self.cache_fit:
            self.cupac_model.fit(pre_experiment_x, pre_experiment_y)
            return

        try:
            check_is_fitted(self.cupac_model)
        except NotFittedError:
            self.cupac_model.fit(pre_experiment_x, pre_experiment_y)

    def _predict_cupac_model(self, df_predict: pd.DataFrame) -> ArrayLike:
        """Predicts the cupac model"""
        if hasattr(self.cupac_model, "predict_proba"):
            return self.cupac_model.predict_proba(df_predict)[:, 1]
        if hasattr(self.cupac_model, "predict"):
            return self.cupac_model.predict(df_predict)
        raise ValueError("cupac_model should have predict or predict_proba method.")

    def need_covariates(self, pre_experiment_df: Optional[pd.DataFrame] = None) -> bool:
        return pre_experiment_df is not None and self.is_cupac

    def check_cupac_inputs(self, pre_experiment_df: Optional[pd.DataFrame] = None):
        if self.is_cupac and pre_experiment_df is None:
            raise ValueError("If cupac is used, pre_experiment_df should be provided.")

        if not self.is_cupac and pre_experiment_df is not None:
            raise ValueError(
                "If cupac is not used, pre_experiment_df should not be provided - remove pre_experiment_df argument or set cupac_model to not None."
            )

    def check_cupac_config(self):
        if self.is_cupac and self.target_col in self.features_cupac_model:
            raise ValueError(
                "If cupac is used, target_col should not be in features_cupac_model."
            )
        if self.is_cupac and self.scale_col in self.features_cupac_model:
            raise ValueError(
                "If cupac is used, scale_col should not be in features_cupac_model."
            )


class MLRateHandler:
    """
    MLRATE variance reducer (Guo et al., NeurIPS 2021).

    Uses K-fold cross-fitting on experiment data to generate ML-predicted
    covariates without overfitting bias. Each unit's prediction comes from
    a model that never saw that unit during training.

    When cluster_cols is provided, splits by cluster (GroupKFold) so no
    cluster appears in both train and validation of the same fold.
    """

    def __init__(
        self,
        ml_model: BaseEstimator,
        n_folds: int = 5,
        target_col: str = "target",
        features: Optional[List[str]] = None,
        cluster_cols: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        self.ml_model = ml_model
        self.n_folds = n_folds
        self.target_col = target_col
        self.features = features or []
        self.cluster_cols = cluster_cols
        self.random_state = random_state

    @property
    def cupac_outcome_name(self) -> str:
        return f"estimate_{self.target_col}"

    def add_covariates(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        feature_cols = self._resolve_feature_cols(df)
        df = df.copy()
        X = df[feature_cols].values
        y = df[self.target_col].values

        splits = self._make_splits(X, y, df)
        df[self.cupac_outcome_name] = self._cross_fit_predict(X, y, splits)
        return df

    def _resolve_feature_cols(self, df: pd.DataFrame) -> List[str]:
        feature_cols = self.features or [c for c in df.columns if c != self.target_col]
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            raise ValueError(
                f"MLRateHandler: features {missing} not found in df columns."
            )
        return feature_cols

    def _make_splits(self, X, y, df: pd.DataFrame):
        if not self.cluster_cols:
            return KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            ).split(X, y)

        groups = df[self.cluster_cols].astype(str).apply("_".join, axis=1).values
        n_unique = len(np.unique(groups))
        if n_unique < self.n_folds:
            raise ValueError(
                f"MLRateHandler: n_folds={self.n_folds} but only {n_unique} unique "
                f"clusters found. Reduce n_folds."
            )
        return GroupKFold(n_splits=self.n_folds).split(X, y, groups=groups)

    def _cross_fit_predict(self, X, y, splits) -> np.ndarray:
        predictions = np.zeros(len(X))
        for train_idx, val_idx in splits:
            model = clone(self.ml_model)
            model.fit(X[train_idx], y[train_idx])
            predictions[val_idx] = model.predict(X[val_idx])
        return predictions


def build_ml_handler(
    cupac_model: Optional[BaseEstimator],
    ml_option: str = "cupac",
    n_folds: int = 5,
    target_col: str = "target",
    features_cupac_model: Optional[List[str]] = None,
    scale_col: Optional[str] = None,
    cluster_cols: Optional[List[str]] = None,
) -> MLHandler:
    """Builds the covariate-injection handler shared by PowerAnalysis and NormalPowerAnalysis."""
    if cupac_model is None:
        return NoOpHandler()
    if ml_option == "mlrate":
        return MLRateHandler(
            ml_model=cupac_model,
            n_folds=n_folds,
            target_col=target_col,
            features=features_cupac_model or [],
            cluster_cols=cluster_cols,
        )
    return CupacHandler(
        cupac_model=cupac_model,
        target_col=target_col,
        scale_col=scale_col,
        features_cupac_model=features_cupac_model,
    )
