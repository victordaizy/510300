"""IF真实期限结构二元筛选V1.0.1的期限选择、时钟和状态测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.if_true_term_structure_binary_screen_v1_0_1 import (  # noqa: E402
    CANDIDATE_CONTRACT_PATH,
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    FEATURE_COLUMNS,
    RANK_COLUMNS,
    align_states_to_market,
    build_candidate_states,
    build_term_features,
    load_candidate_contract,
    sha256_file,
)


def _synthetic_inputs(days: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=days)
    symbols = ["IF2401", "IF2402", "IF2404"]
    expiries = pd.to_datetime(["2024-01-19", "2024-02-16", "2024-04-19"])
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for contract_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": 101.0 + day_index + contract_index,
                    "open_interest": float((contract_index + 1) * 10),
                    "source_url": "https://example.invalid/cffex.zip",
                }
            )
    daily = pd.DataFrame(rows)
    expiry = pd.DataFrame(
        {
            "symbol": symbols,
            "expiry_date": expiries,
            "expiry_source": ["CFFEX_HISTORY_LAST_APPEARANCE"] * 3,
            "active_at_ceiling": [False, False, False],
        }
    )
    spot = pd.DataFrame(
        {
            "date": dates,
            "close": 100.0 + np.arange(days, dtype=float),
            "symbol": "000300.SH",
        }
    )
    return daily, expiry, spot


def test_true_near_next_far_selection_and_formulas_are_exact() -> None:
    """近月、次月、最远月和四个年化变量必须按冻结公式计算。"""

    daily, expiry, spot = _synthetic_inputs()
    eligible, features, audit = build_term_features(
        daily,
        expiry,
        spot,
        minimum_dte=5,
        minimum_contracts=3,
        rolling_window=3,
        minimum_observations=1,
    )
    first = features.iloc[0]
    expected_contract_basis = np.array(
        [
            np.log(101.0 / 100.0) * 365.0 / 17.0,
            np.log(102.0 / 100.0) * 365.0 / 45.0,
            np.log(103.0 / 100.0) * 365.0 / 108.0,
        ]
    )

    assert first["near_symbol"] == "IF2401"
    assert first["next_symbol"] == "IF2402"
    assert first["far_symbol"] == "IF2404"
    assert np.isclose(first["annualized_near_basis"], expected_contract_basis[0])
    assert np.isclose(
        first["annualized_near_next_curve"],
        np.log(102.0 / 101.0) * 365.0 / 28.0,
    )
    assert np.isclose(
        first["annualized_near_far_curve"],
        np.log(103.0 / 101.0) * 365.0 / 91.0,
    )
    assert np.isclose(
        first["annualized_oi_weighted_basis"],
        np.average(expected_contract_basis, weights=[10.0, 20.0, 30.0]),
    )
    assert len(eligible) == 18
    assert audit["minimum_eligible_contract_count"] == 3
    assert audit["future_price_or_return_columns_read"] == 0


def test_future_contract_prices_cannot_change_prior_term_features() -> None:
    """改写最后一日逐合约价格不得改变此前期限结构或历史分位。"""

    daily, expiry, spot = _synthetic_inputs()
    changed = daily.copy()
    last_date = changed["date"].max()
    changed.loc[changed["date"].eq(last_date), "close"] *= [0.8, 1.2, 1.5]
    _, first, _ = build_term_features(
        daily,
        expiry,
        spot,
        minimum_dte=5,
        minimum_contracts=3,
        rolling_window=3,
        minimum_observations=2,
    )
    _, second, _ = build_term_features(
        changed,
        expiry,
        spot,
        minimum_dte=5,
        minimum_contracts=3,
        rolling_window=3,
        minimum_observations=2,
    )

    pdt.assert_frame_equal(
        first.loc[first.index[:-1], [*FEATURE_COLUMNS, *RANK_COLUMNS]],
        second.loc[second.index[:-1], [*FEATURE_COLUMNS, *RANK_COLUMNS]],
    )


def test_eight_candidate_states_match_frozen_upper_and_lower_tails() -> None:
    """四个变量的上下尾规则必须严格采用10%和90%阈值。"""

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            **{column: [0.0, -0.1, 0.1] for column in FEATURE_COLUMNS},
            "rank_annualized_near_basis": [np.nan, 0.10, 0.90],
            "rank_annualized_near_next_curve": [np.nan, 0.10, 0.90],
            "rank_annualized_near_far_curve": [np.nan, 0.10, 0.90],
            "rank_annualized_oi_weighted_basis": [np.nan, 0.10, 0.90],
            "near_symbol": ["IF2401"] * 3,
            "next_symbol": ["IF2402"] * 3,
            "far_symbol": ["IF2404"] * 3,
        }
    )
    states = build_candidate_states(features)

    assert states.loc[0, EXPECTED_CANDIDATES].tolist() == [1] * 8
    assert states.loc[1, EXPECTED_CANDIDATES].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert states.loc[2, EXPECTED_CANDIDATES].tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8


def test_missing_term_feature_market_day_defaults_to_full_state() -> None:
    """交易日没有真实期限结构时不得填值触发空仓。"""

    daily, expiry, spot = _synthetic_inputs()
    _, features, _ = build_term_features(
        daily,
        expiry,
        spot,
        minimum_dte=5,
        minimum_contracts=3,
        rolling_window=3,
        minimum_observations=1,
    )
    states = build_candidate_states(features)
    market = pd.DataFrame(
        {"trade_date": [features.loc[0, "date"], pd.Timestamp("2024-01-13")]}
    )
    aligned = align_states_to_market(market, states)

    assert not bool(aligned.loc[1, "term_feature_observed"])
    assert aligned.loc[1, EXPECTED_CANDIDATES].tolist() == [1] * 8


def test_candidate_contract_hash_rules_and_double_twenty_gate_are_immutable() -> None:
    """正式获取前合同、八规则、二元状态和双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(CANDIDATE_CONTRACT_PATH) == CANDIDATE_CONTRACT_SHA256
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
