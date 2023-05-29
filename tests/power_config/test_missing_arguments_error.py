import pytest

from cluster_experiments.power_config import MissingArgumentError, PowerConfig


def test_missing_argument_segment_cols(caplog):
    msg = "segment_cols is required when using perturbator = segmented_beta_relative."
    with pytest.raises(MissingArgumentError, match=msg):
        PowerConfig(
            cluster_cols=["cluster"],
            analysis="mlm",
            perturbator="segmented_beta_relative",
            splitter="non_clustered",
            n_simulations=4,
            average_effect=1.5,
        )
