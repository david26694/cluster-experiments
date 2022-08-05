from abc import abstractmethod

import pandas as pd


class Perturbator:
    @abstractmethod
    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class UniformPerturbator:
    def __init__(
        self,
        average_effect: float,
        target: str,
        treatment_col: str,
        treatment: str = "B",
    ):
        self.average_effect = average_effect
        self.target = target
        self.treatment_col = treatment_col
        self.treatment = treatment
        self.treated_query = f"{self.treatment_col} == {self.treatment}"

    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df[self.treated_query], self.target] += self.average_effect
        return df


class BinaryPerturbator:
    def __init__(
        self,
        average_effect: float,
        target: str,
        treatment_col: str,
        treatment: str = "B",
    ):
        self.average_effect = average_effect
        self.target = target
        self.treatment_col = treatment_col
        self.treatment = treatment
        self.treated_query = f"{self.treatment_col} == {self.treatment}"

    def _sample_max(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Like sample without replacement,
        but if you are to sample more than 100% of the data,
        it just returns the whole dataframe."""
        if n >= len(df):
            return df
        return df.sample(n=n)

    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        from_target, to_target = 1, 0
        if self.average_effect > 0:
            from_target, to_target = 0, 1

        n_transformed = abs(
            int(self.average_effect * len(df.query(self.treated_query)))
        )
        idx = (
            # Sample of negative cases in group B
            df.query(f"{self.target} == {from_target} & {self.treated_query}")
            .pipe(self._sample_max, n=n_transformed)
            .index.drop_duplicates()
        )
        df.loc[idx, self.target] = to_target
        return df
