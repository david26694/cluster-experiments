"""
Pins the power and MDE numbers for absolute effects.

Relative effects were introduced by reworking the shared machinery: power and the
MDE are now both derived from a :class:`StandardErrorCurve` rather than from a
scalar standard error. That rework was supposed to leave absolute effects
completely alone, which is easy to claim and easy to break, since every absolute
analysis flows through the same rewritten code.

The expected values below were captured by running this file against the code on
``main``, before that rework. A failure here means an absolute-effect number moved,
which is a regression unless it was deliberate.

Only closed-form estimators are covered. GEE and MLM fit iteratively, so their
output shifts with the linear-algebra backend and would make this brittle rather
than informative.
"""

import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    DeltaMethodAnalysis,
    OLSAnalysis,
)
from cluster_experiments.power_analysis import NormalPowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter, NonClusteredSplitter

# A relative tolerance rather than exact equality: the arithmetic is identical, but
# the last bit or two can differ across BLAS builds, and a spurious CI failure
# would get this test deleted rather than investigated. 1e-12 is still far tighter
# than any formula change could hide in.
TOLERANCE = 1e-12

N_USERS = 400
N_DAYS = 28
SEED = 42
N_SIMULATIONS = 5
POWERS = [0.6, 0.8, 0.95]
EFFECTS = [0.05, 0.2, 0.5]
LENGTHS = [7, 14, 28]
HYPOTHESES = ["two-sided", "greater", "less"]


def _flat_df() -> pd.DataFrame:
    """One row per (user, day), for the regression analyses."""
    rng = np.random.default_rng(SEED)
    users = np.repeat(np.arange(N_USERS), N_DAYS)
    days = np.tile(np.arange(N_DAYS), N_USERS)
    user_effect = rng.normal(0, 2, N_USERS)[users]
    return pd.DataFrame(
        {
            "user": users,
            "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(days, unit="D"),
            "target": 10 + user_effect + rng.normal(0, 3, N_USERS * N_DAYS),
        }
    )


def _aggregated_df() -> pd.DataFrame:
    """One row per user with a scale column, for the delta method."""
    rng = np.random.default_rng(SEED + 1)
    scale = rng.uniform(5, 25, N_USERS)
    return pd.DataFrame(
        {
            "user": np.arange(N_USERS),
            "target": scale * rng.normal(2.0, 0.4, N_USERS),
            "scale": scale,
        }
    )


def _power_analysis(analysis, clustered: bool, time_col=None) -> NormalPowerAnalysis:
    return NormalPowerAnalysis(
        analysis=analysis,
        splitter=(
            ClusteredSplitter(cluster_cols=["user"])
            if clustered
            else NonClusteredSplitter()
        ),
        n_simulations=N_SIMULATIONS,
        seed=SEED,
        time_col=time_col,
    )


def _ols(hypothesis):
    return OLSAnalysis(target_col="target", hypothesis=hypothesis)


def _clustered_ols(hypothesis):
    return ClusteredOLSAnalysis(
        cluster_cols=["user"], target_col="target", hypothesis=hypothesis
    )


def _delta(hypothesis):
    return DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        hypothesis=hypothesis,
    )


# case name -> (analysis factory, uses a clustered splitter, uses the time column)
CASES = {
    "ols": (_ols, False, False),
    "clustered_ols": (_clustered_ols, True, True),
    "delta": (_delta, True, False),
}


def _measure(case: str, hypothesis: str) -> dict:
    """Every public power and MDE number for one analysis and hypothesis."""
    build, clustered, uses_time = CASES[case]
    df = _aggregated_df() if case == "delta" else _flat_df()
    power_analysis = _power_analysis(
        build(hypothesis), clustered, time_col="date" if uses_time else None
    )

    measured = {
        "mde_power_line": [
            power_analysis.mde_power_line(df, powers=POWERS)[power] for power in POWERS
        ],
        "power_line": [
            power_analysis.power_line(df, average_effects=EFFECTS)[effect]
            for effect in EFFECTS
        ],
    }
    if uses_time:
        measured["mde_time_line"] = [
            row["mde"]
            for row in power_analysis.mde_time_line(
                df, powers=POWERS, experiment_length=LENGTHS
            )
        ]
        measured["power_time_line"] = [
            row["power"]
            for row in power_analysis.power_time_line(
                df, average_effects=EFFECTS, experiment_length=LENGTHS
            )
        ]
        measured["mde_rolling_time_line"] = [
            row["mde"]
            for row in power_analysis.mde_rolling_time_line(
                df, powers=POWERS, experiment_length=LENGTHS, agg_func="sum"
            )
        ]
    return measured


# Captured against main. Regenerate with `python <this file>` after an intentional
# change, and say so in the commit message.
EXPECTED: dict = {
    "clustered_ols|two-sided": {
        "mde_power_line": [0.4530675421678498, 0.573834656754106, 0.739920174586466],
        "power_line": [0.05686740287083553, 0.16408598379443895, 0.6852964542593956],
        "mde_time_line": [
            0.5102991883837502,
            0.6459311892266342,
            0.8311237908030488,
            0.4763608717659663,
            0.6029724353969304,
            0.7758484876026158,
            0.4529307371768927,
            0.5733148246396127,
            0.7376878502314814,
        ],
        "power_time_line": [
            0.055415320936746386,
            0.13983025339761906,
            0.583489960921655,
            0.05619042938359055,
            0.15301789008544928,
            0.6407413185836545,
            0.05683340936871696,
            0.16398428207677138,
            0.6834984526427779,
        ],
        "mde_rolling_time_line": [
            3.9971752817328974,
            5.059581206575272,
            6.510195485868187,
            7.152488103101428,
            9.053542023059931,
            11.64925040298705,
            12.71374827136772,
            16.092924949504237,
            20.70686941925642,
        ],
    },
    "clustered_ols|greater": {
        "mde_power_line": [0.3885640586678386, 0.509292181759074, 0.67524103241359],
        "power_line": [0.08068103826263107, 0.2516044783224666, 0.7874651687913004],
        "mde_time_line": [
            0.43764760288176835,
            0.5732796037246524,
            0.758472205301067,
            0.40854110369124647,
            0.5351526673222106,
            0.7080287195278959,
            0.3884467306812035,
            0.5088308181439235,
            0.6732038437357922,
        ],
        "power_time_line": [
            0.07667775652986775,
            0.21871078384790943,
            0.7005190090711884,
            0.07885497167885247,
            0.2367791155868706,
            0.7503293113193654,
            0.08059083868462524,
            0.25146953967699925,
            0.7859943185451665,
        ],
        "mde_rolling_time_line": [
            3.428095164895946,
            4.490501089738321,
            5.941115369031236,
            6.134184306420476,
            8.03523822637898,
            10.630946606306098,
            10.903684703535168,
            14.282861381671687,
            18.89680585142387,
        ],
    },
    "clustered_ols|less": {
        "mde_power_line": [
            -0.38856405866783883,
            -0.5092921817590742,
            -0.6752410324135902,
        ],
        "power_line": [
            0.029432382235226023,
            0.004393162996090431,
            2.1815106469320516e-05,
        ],
        "mde_time_line": [
            -0.4376476028817685,
            -0.5732796037246526,
            -0.7584722053010672,
            -0.40854110369124663,
            -0.5351526673222107,
            -0.7080287195278961,
            -0.38844673068120367,
            -0.5088308181439236,
            -0.6732038437357925,
        ],
        "power_time_line": [
            0.031306593634351,
            0.005982878068026952,
            6.79259158934161e-05,
            0.03026647408786294,
            0.005040978206173504,
            3.6662041514299894e-05,
            0.029472796472083347,
            0.004398619026653175,
            2.2295469897662992e-05,
        ],
        "mde_rolling_time_line": [
            -3.4280951648959475,
            -4.490501089738323,
            -5.9411153690312375,
            -6.134184306420479,
            -8.035238226378981,
            -10.6309466063061,
            -10.903684703535173,
            -14.282861381671692,
            -18.896805851423874,
        ],
    },
    "delta|two-sided": {
        "mde_power_line": [
            0.09513236710836329,
            0.12070657679521105,
            0.15521536735836772,
        ],
        "power_line": [0.21372096220160822, 0.996190755539734, 1.0],
    },
    "delta|greater": {
        "mde_power_line": [
            0.08158831793033157,
            0.10713001580705306,
            0.14164742157502988,
        ],
        "power_line": [0.31506595398377824, 0.9985757744254843, 1.0],
    },
    "delta|less": {
        "mde_power_line": [
            -0.08158831793033161,
            -0.1071300158070531,
            -0.1416474215750299,
        ],
        "power_line": [
            0.0024912263750688475,
            1.7668796624386196e-10,
            1.475633368909723e-40,
        ],
    },
    "ols|two-sided": {
        "mde_power_line": [
            0.15091028250748656,
            0.19103091169496259,
            0.2457717320768557,
        ],
        "power_line": [0.11350597320539038, 0.8348171250395717, 0.9999999614481165],
    },
    "ols|greater": {
        "mde_power_line": [
            0.1294250998101628,
            0.16954456942506088,
            0.22428792159695196,
        ],
        "power_line": [0.18099253101246104, 0.9012115449471072, 0.9999999936065497],
    },
    "ols|less": {
        "mde_power_line": [
            -0.12942509981016287,
            -0.16954456942506094,
            -0.22428792159695204,
        ],
        "power_line": [
            0.00870062458139587,
            2.3450487674163306e-06,
            1.3712233876228248e-19,
        ],
    },
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("hypothesis", HYPOTHESES)
def test_absolute_effect_numbers_are_unchanged(case, hypothesis):
    """Absolute-effect power and MDE must match the values captured from main."""
    key = f"{case}|{hypothesis}"
    assert key in EXPECTED, f"no golden values for {key}; regenerate this file"

    measured = _measure(case, hypothesis)
    expected = EXPECTED[key]

    assert sorted(measured) == sorted(
        expected
    ), f"{key}: the set of reported quantities changed"
    for method, values in measured.items():
        assert values == pytest.approx(
            expected[method], rel=TOLERANCE
        ), f"{key}.{method} moved"
