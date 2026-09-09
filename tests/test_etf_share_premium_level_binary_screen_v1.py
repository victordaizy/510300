"""510300 ETF份额与折溢价水平二元筛选V1的时钟、分位与状态测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.etf_share_premium_level_binary_screen_v1 import (  # noqa: E402
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    align_states_to_market,
    build_candidate_states,
    build_features,
    load_candidate_contract,
    rolling_last_percentile,
    sha256_file,
)


def _raw_features(rows: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    return pd.DataFrame(
        {
            "date": dates,
            "fund_shares": np.linspace(10_000_000_000.0, 11_000_000_000.0, rows),
            "close_premium_to_nav": np.linspace(-0.002, 0.002, rows),
            "feature_asof": dates + pd.Timedelta(hours=15),
            "execution_earliest": "NEXT_TRADING_DAY_OPEN",
        }
    )


def test_rolling_percentile_uses_only_current_and_past_finite_values() -> None:
    """当前经验分位不得读取未来值，且使用小于等于定义。"""

    values = pd.Series([4.0, np.nan, 2.0, 3.0, 1.0, 5.0])
    result = rolling_last_percentile(values, window=4, minimum_observations=3)

    assert result.iloc[:3].isna().all()
    assert np.isclose(result.iloc[3], 2.0 / 3.0)
    assert np.isclose(result.iloc[4], 1.0 / 3.0)
    assert np.isclose(result.iloc[5], 1.0)


def test_future_observation_cannot_change_prior_features() -> None:
    """改写最后一日输入不得改变此前任何份额或折溢价特征。"""

    original = _raw_features()
    changed = original.copy()
    changed.loc[changed.index[-1], "fund_shares"] *= 5.0
    changed.loc[changed.index[-1], "close_premium_to_nav"] = -0.05
    first = build_features(original, rolling_window=4, minimum_observations=3)
    second = build_features(changed, rolling_window=4, minimum_observations=3)
    feature_columns = [
        "share_change_1d",
        "share_change_5d",
        "share_change_20d",
        "share_rank_1d",
        "share_rank_5d",
        "share_rank_20d",
        "premium_rank",
        "absolute_premium_rank",
    ]

    pdt.assert_frame_equal(
        first.loc[first.index[:-1], feature_columns],
        second.loc[second.index[:-1], feature_columns],
    )


def test_eight_candidate_states_match_frozen_thresholds() -> None:
    """八条规则必须严格采用固定阈值，缺失特征保持满仓。"""

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "share_change_1d": [np.nan, -0.01, 0.01],
            "share_change_5d": [np.nan, -0.02, 0.02],
            "share_change_20d": [np.nan, -0.03, 0.03],
            "close_premium_to_nav": [0.0, -0.002, 0.002],
            "absolute_premium_level": [0.0, 0.002, 0.002],
            "share_rank_1d": [np.nan, 0.10, 0.50],
            "share_rank_5d": [np.nan, 0.11, 0.80],
            "share_rank_20d": [np.nan, 0.10, 0.50],
            "premium_rank": [np.nan, 0.10, 0.80],
            "absolute_premium_rank": [np.nan, 0.90, 0.50],
        }
    )
    states = build_candidate_states(features)

    assert states.loc[0, EXPECTED_CANDIDATES].tolist() == [1] * 8
    assert states.loc[1, EXPECTED_CANDIDATES].tolist() == [0, 1, 0, 0, 1, 0, 0, 1]
    assert states.loc[2, EXPECTED_CANDIDATES].tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8
        assert set(states[candidate_id].unique()).issubset({0, 1})


def test_missing_joint_feature_market_day_defaults_to_full_state() -> None:
    """交易日没有联合特征时不得填值触发空仓。"""

    features = build_features(_raw_features(6), rolling_window=3, minimum_observations=2)
    states = build_candidate_states(features)
    market_dates = pd.DataFrame(
        {
            "trade_date": pd.DatetimeIndex(
                [features.loc[0, "date"], pd.Timestamp("2024-01-06")]
            )
        }
    )
    aligned = align_states_to_market(market_dates, states)

    assert not bool(aligned.loc[1, "feature_observed"])
    assert aligned.loc[1, EXPECTED_CANDIDATES].tolist() == [1] * 8


def test_candidate_contract_hash_and_double_twenty_gate_are_immutable() -> None:
    """全历史合并前合同、八规则、二元状态和双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(CANDIDATE_CONTRACT_PATH := ROOT / "config" / "510300_etf_share_premium_level_binary_screen_v1_candidates.yaml") == CANDIDATE_CONTRACT_SHA256
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
