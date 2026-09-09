"""中金所IF会员持仓二元筛选V1的公式、时钟、覆盖和状态测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cffex_member_position_binary_screen_v1 import (  # noqa: E402
    CANDIDATE_CONTRACT_PATH,
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    FEATURE_COLUMNS,
    RANK_COLUMNS,
    align_states_to_market,
    build_candidate_states,
    build_member_features,
    load_candidate_contract,
    sha256_file,
)


def _synthetic_inputs(days: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=days)
    symbols = ["IF2401", "IF2402", "IF2404"]
    expiries = pd.to_datetime(["2024-01-19", "2024-02-16", "2024-04-19"])
    rows: list[dict[str, object]] = []
    for day_index, date in enumerate(dates):
        for contract_index, symbol in enumerate(symbols):
            for rank in range(1, 21):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "rank": rank,
                        "long_open_interest": 100 * (contract_index + 1)
                        + rank
                        + 3 * day_index,
                        "long_open_interest_change": contract_index
                        + rank
                        - 10
                        + day_index,
                        "short_open_interest": 80 * (contract_index + 1)
                        + 2 * rank
                        + day_index,
                        "short_open_interest_change": contract_index
                        - rank
                        + 5
                        - day_index,
                        "source_url": "https://example.invalid/cffex.csv",
                        "raw_file_sha256": "0" * 64,
                    }
                )
    expiry = pd.DataFrame(
        {
            "symbol": symbols,
            "expiry_date": expiries,
            "expiry_source": ["CFFEX_HISTORY_LAST_APPEARANCE"] * 3,
            "active_at_ceiling": [False, False, False],
        }
    )
    return pd.DataFrame(rows), expiry


def _build(
    ranks: pd.DataFrame,
    expiry: pd.DataFrame,
    *,
    minimum_observations: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    return build_member_features(
        ranks,
        expiry,
        minimum_dte=5,
        minimum_rank_rows=5,
        maximum_rank=20,
        minimum_contracts=3,
        rolling_window=3,
        minimum_observations=minimum_observations,
    )


def test_member_position_aggregations_and_formulas_are_exact() -> None:
    """跨期限Top20、近月净比率和Top5集中度差必须按冻结公式计算。"""

    ranks, expiry = _synthetic_inputs()
    summaries, features, audit = _build(ranks, expiry)
    first_date = ranks["date"].min()
    first_raw = ranks.loc[ranks["date"].eq(first_date)]
    near_raw = first_raw.loc[first_raw["symbol"].eq("IF2401")]
    all_long = float(first_raw["long_open_interest"].sum())
    all_short = float(first_raw["short_open_interest"].sum())
    near_long = float(near_raw["long_open_interest"].sum())
    near_short = float(near_raw["short_open_interest"].sum())
    long_top5 = float(
        first_raw.loc[first_raw["rank"].le(5), "long_open_interest"].sum()
    )
    short_top5 = float(
        first_raw.loc[first_raw["rank"].le(5), "short_open_interest"].sum()
    )
    first = features.iloc[0]

    assert first["near_symbol"] == "IF2401"
    assert first["eligible_contract_count"] == 3
    assert np.isclose(
        first["all_top20_net_ratio"], (all_long - all_short) / (all_long + all_short)
    )
    assert np.isclose(
        first["all_top20_net_change_ratio"],
        (
            float(first_raw["long_open_interest_change"].sum())
            - float(first_raw["short_open_interest_change"].sum())
        )
        / (all_long + all_short),
    )
    assert np.isclose(
        first["near_top20_net_ratio"],
        (near_long - near_short) / (near_long + near_short),
    )
    assert np.isclose(
        first["all_top5_concentration_spread"],
        long_top5 / all_long - short_top5 / all_short,
    )
    assert first["feature_asof"] == pd.Timestamp(first_date) + pd.Timedelta(
        hours=16, minutes=30
    )
    assert first["execution_earliest"] == "NEXT_TRADING_DAY_OPEN"
    assert len(summaries) == 18
    assert audit["feature_rows"] == 6
    assert audit["minimum_eligible_contract_count"] == 3
    assert audit["future_price_or_return_columns_read"] == 0


def test_future_member_positions_cannot_change_prior_features() -> None:
    """改写最后一日会员持仓不得改变此前特征或历史分位。"""

    ranks, expiry = _synthetic_inputs()
    changed = ranks.copy()
    last_date = changed["date"].max()
    mask = changed["date"].eq(last_date)
    changed.loc[mask, "long_open_interest"] *= 3
    changed.loc[mask, "short_open_interest"] += 777
    changed.loc[mask, "long_open_interest_change"] -= 500
    changed.loc[mask, "short_open_interest_change"] += 500
    _, first, _ = _build(ranks, expiry, minimum_observations=2)
    _, second, _ = _build(changed, expiry, minimum_observations=2)

    pdt.assert_frame_equal(
        first.loc[first.index[:-1], [*FEATURE_COLUMNS, *RANK_COLUMNS]],
        second.loc[second.index[:-1], [*FEATURE_COLUMNS, *RANK_COLUMNS]],
    )


def test_eight_candidate_states_match_frozen_upper_and_lower_tails() -> None:
    """四个会员变量的上下尾规则必须严格采用10%和90%阈值。"""

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            **{column: [0.0, -0.1, 0.1] for column in FEATURE_COLUMNS},
            **{
                column: [np.nan, 0.10, 0.90]
                for column in RANK_COLUMNS
            },
            "near_symbol": ["IF2401"] * 3,
        }
    )
    states = build_candidate_states(features)

    assert states.loc[0, EXPECTED_CANDIDATES].tolist() == [1] * 8
    assert states.loc[1, EXPECTED_CANDIDATES].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert states.loc[2, EXPECTED_CANDIDATES].tolist() == [1, 0, 1, 0, 1, 0, 1, 0]
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8


def test_fewer_than_three_contracts_creates_no_feature_and_defaults_full() -> None:
    """不得把至少3个合约的固定门槛降到2个，缺失交易日必须保持满仓。"""

    ranks, expiry = _synthetic_inputs()
    first_date = ranks["date"].min()
    ranks = ranks.loc[
        ~(ranks["date"].eq(first_date) & ranks["symbol"].eq("IF2404"))
    ].copy()
    _, features, audit = _build(ranks, expiry)
    assert first_date not in set(features["date"])
    assert first_date.date().isoformat() in audit["insufficient_contract_dates_excluded"]
    states = build_candidate_states(features)
    market = pd.DataFrame({"trade_date": [first_date, features.loc[0, "date"]]})
    aligned = align_states_to_market(market, states)

    assert not bool(aligned.loc[0, "member_feature_observed"])
    assert aligned.loc[0, EXPECTED_CANDIDATES].tolist() == [1] * 8


def test_candidate_contract_hash_rules_and_double_twenty_gate_are_immutable() -> None:
    """获取前合同、八规则、二元状态、至少3合约和双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(CANDIDATE_CONTRACT_PATH) == CANDIDATE_CONTRACT_SHA256
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
    assert (
        contract["contract_and_rank_selection"]["minimum_eligible_contracts_per_day"]
        == 3
    )
