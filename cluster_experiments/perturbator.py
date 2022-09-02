from abc import abstractmethod

import pandas as pd


class Perturbator:
    def __init__(
        self,
        average_effect: float,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
    ):
        self.average_effect = average_effect
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.treatment = treatment
        self.treated_query = f"{self.treatment_col} == '{self.treatment}'"

    @abstractmethod
    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @classmethod
    def from_dict(cls, config: dict) -> "Perturbator":
        return cls(**config)

    @classmethod
    def from_config(cls, config):
        return cls(
            average_effect=config.average_effect,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
        )


class UniformPerturbator(Perturbator):
    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df.loc[
            df[self.treatment_col] == self.treatment, self.target_col
        ] += self.average_effect
        return df


class BinaryPerturbator(Perturbator):
    def _sample_max(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Like sample without replacement,
        but if you are to sample more than 100% of the data,
        it just returns the whole dataframe."""
        if n >= len(df):
            return df
        return df.sample(n=n)

    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        from_target, to_target = 1, 0
        if self.average_effect > 0:
            from_target, to_target = 0, 1

        n_transformed = abs(
            int(self.average_effect * len(df.query(self.treated_query)))
        )
        idx = list(
            # Sample of negative cases in group B
            df.query(f"{self.target_col} == {from_target} & {self.treated_query}")
            .pipe(self._sample_max, n=n_transformed)
            .index.drop_duplicates()
        )
        df.loc[idx, self.target_col] = to_target
        return df
