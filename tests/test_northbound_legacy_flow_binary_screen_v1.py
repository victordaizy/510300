"""北向旧口径流量二元筛选V1的因果性与协议回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.northbound_legacy_flow_binary_screen_v1 import (  # noqa: E402
    align_states_to_market,
    build_candidate_states,
    load_config,
    rolling_last_percentile,
)


def test_rolling_percentile_is_not_changed_by_future_observation() -> None:
    """追加未来极端值不得改写任何既有日期的滚动分位。"""

    history = pd.Series([3.0, 1.0, 4.0, 2.0, 5.0, -1.0], dtype=float)
    original = rolling_last_percentile(
        history,
        window=4,
        minimum_observations=2,
    )
    extended = rolling_last_percentile(
        pd.concat([history, pd.Series([-1_000_000.0])], ignore_index=True),
        window=4,
        minimum_observations=2,
    )

    pd.testing.assert_series_equal(
        original.reset_index(drop=True),
        extended.iloc[: len(history)].reset_index(drop=True),
    )


def test_candidate_rules_generate_the_six_exact_binary_states() -> None:
    """用边界分位验证六条预先固定规则的精确0/1映射。"""

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "flow_1d": [-1.0, 1.0, 0.0],
            "flow_5d": [-5.0, 5.0, 0.0],
            "flow_20d": [-20.0, 20.0, 0.0],
            "rank_1d": [0.10, 0.11, np.nan],
            "rank_5d": [0.21, 0.10, 0.20],
            "rank_20d": [0.10, 0.11, np.nan],
        }
    )

    states = build_candidate_states(features)

    assert states["NB_FLOW_SIGN"].tolist() == [0, 1, 1]
    assert states["NB1D_BOTTOM10_CASH"].tolist() == [0, 1, 1]
    assert states["NB5D_BOTTOM10_CASH"].tolist() == [1, 0, 1]
    assert states["NB20D_BOTTOM10_CASH"].tolist() == [0, 1, 1]
    assert states["NB1D_OR_5D_BOTTOM10_CASH"].tolist() == [0, 0, 1]
    assert states["NB1D_AND_5D_BOTTOM20_CASH"].tolist() == [1, 0, 1]
    for column in [item for item in states.columns if item.startswith("NB")]:
        assert states[column].dtype == np.int8
        assert set(states[column].unique()).issubset({0, 1})


def test_market_day_without_flow_disclosure_returns_to_full_position() -> None:
    """互联互通无新披露日不得延续旧空仓信号。"""

    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-02-08", "2024-02-19", "2024-02-20"]
            )
        }
    )
    candidate_states = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-08", "2024-02-20"]),
            "TEST_RULE": np.array([0, 0], dtype=np.int8),
        }
    )

    aligned = align_states_to_market(market, candidate_states, ["TEST_RULE"])

    assert aligned["flow_observed"].tolist() == [True, False, True]
    assert aligned["TEST_RULE"].tolist() == [0, 1, 0]


def test_protocol_scope_and_double_twenty_gate_are_fixed() -> None:
    """协议必须固定六候选、两种持仓状态与双20硬门。"""

    config = load_config()

    assert config["protocol"]["frozen_candidate_family_size"] == 6
    assert len(config["candidates"]) == 6
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["allowed_target_states"] == [0, 1]
    assert config["objective"]["minimum_annualized_excess"] == 0.20
    assert config["objective"]["minimum_rolling_excess_median"] == 0.20
    assert config["protocol"]["no_parameter_rescue"] is True
