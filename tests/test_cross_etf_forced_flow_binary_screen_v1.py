"""跨ETF被迫资金流二元筛选V1的宇宙、时钟、特征和状态测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cross_etf_forced_flow_binary_screen_v1 import (  # noqa: E402
    CANDIDATE_CONTRACT_PATH,
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    align_weekly_states_to_daily_targets,
    build_candidate_weekly_states,
    build_weekly_features,
    load_candidate_contract,
    rolling_last_percentile,
    sha256_file,
)
from scripts.collect_510300_cross_etf_forced_flow_sse_weekly_v1 import (  # noqa: E402
    build_frozen_universe,
    load_frozen_contract,
)


def _synthetic_inputs(rows: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-05", periods=rows, freq="7D")
    share_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        share_rows.extend(
            [
                {
                    "date": date,
                    "ts_code": "510300.SH",
                    "fund_shares": 100.0 + 5.0 * index,
                },
                {
                    "date": date,
                    "ts_code": "510050.SH",
                    "fund_shares": 200.0 + (-1.0) ** index * 2.0 * index,
                },
            ]
        )
        price_rows.extend(
            [
                {"date": date, "con_code": "510300.SH", "raw_close": 4.0},
                {"date": date, "con_code": "510050.SH", "raw_close": 3.0},
            ]
        )
    return pd.DataFrame(share_rows), pd.DataFrame(price_rows)


def test_candidate_contract_hash_and_double_twenty_gate_are_immutable() -> None:
    """采集前候选合同、五规则、二元状态和双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(CANDIDATE_CONTRACT_PATH) == CANDIDATE_CONTRACT_SHA256
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
    assert contract["information_clock"]["execution_time"] == "T_PLUS_2_OPEN"


def test_frozen_universe_rebuilds_from_pre_2021_inputs_only() -> None:
    """大ETF池必须由冻结的2020年信息重建为预先记录的10只。"""

    contract, receipt = load_frozen_contract()
    symbols, diagnostics, _ = build_frozen_universe(contract, receipt)

    assert symbols == sorted(contract["universe_freeze"]["broad_pool_expected_symbols"])
    assert len(symbols) == 10
    assert diagnostics["return_overlap_days"].ge(200).all()
    assert diagnostics["h00300_return_correlation"].ge(0.80).all()
    assert diagnostics["average_daily_amount_cny"].ge(100_000_000.0).all()
    assert diagnostics["last_trade_date"].le(pd.Timestamp("2020-12-31")).all()


def test_rolling_percentile_uses_only_current_and_past_finite_values() -> None:
    """当前经验分位不得读取未来值，并严格使用小于等于定义。"""

    values = pd.Series([4.0, np.nan, 2.0, 3.0, 1.0, 5.0])
    result = rolling_last_percentile(values, window=4, minimum_observations=3)

    assert result.iloc[:3].isna().all()
    assert np.isclose(result.iloc[3], 2.0 / 3.0)
    assert np.isclose(result.iloc[4], 1.0 / 3.0)
    assert np.isclose(result.iloc[5], 1.0)


def test_weekly_flow_is_share_change_times_current_close_over_prior_aum() -> None:
    """周度资金流必须由份额变化乘同日价格并除以前周规模形成。"""

    dates = pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"])
    shares = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ts_code": ["510300.SH", "510050.SH"] * 3,
            "fund_shares": [100.0, 100.0, 110.0, 100.0, 90.0, 120.0],
        }
    )
    prices = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "con_code": ["510300.SH", "510050.SH"] * 3,
            "raw_close": [1.0] * 6,
        }
    )
    features = build_weekly_features(
        shares,
        prices,
        broad_symbols=["510300.SH", "510050.SH"],
        hs300_symbols=["510300.SH"],
        rolling_window=3,
        minimum_observations=2,
    )

    assert np.isnan(features.loc[0, "broad_flow_ratio_1w"])
    assert np.isclose(features.loc[1, "broad_flow_ratio_1w"], 10.0 / 200.0)
    assert np.isclose(features.loc[2, "broad_flow_ratio_1w"], 0.0 / 210.0)
    assert np.isclose(features.loc[1, "hs300_flow_ratio_1w"], 10.0 / 100.0)
    assert np.isclose(features.loc[2, "hs300_flow_ratio_1w"], -20.0 / 110.0)


def test_future_share_mutation_cannot_change_prior_features() -> None:
    """改写最后一周份额不得改变此前任何周度流量或分位特征。"""

    shares, prices = _synthetic_inputs()
    changed = shares.copy()
    final_date = changed["date"].max()
    changed.loc[changed["date"].eq(final_date), "fund_shares"] *= 5.0
    first = build_weekly_features(
        shares,
        prices,
        broad_symbols=["510300.SH", "510050.SH"],
        hs300_symbols=["510300.SH"],
        rolling_window=4,
        minimum_observations=3,
    )
    second = build_weekly_features(
        changed,
        prices,
        broad_symbols=["510300.SH", "510050.SH"],
        hs300_symbols=["510300.SH"],
        rolling_window=4,
        minimum_observations=3,
    )
    columns = [
        "broad_flow_ratio_1w",
        "broad_flow_ratio_4w",
        "hs300_flow_ratio_1w",
        "broad_flow_rank_1w",
        "broad_flow_rank_4w",
        "hs300_flow_rank_1w",
    ]

    pdt.assert_frame_equal(first.iloc[:-1][columns], second.iloc[:-1][columns])


def test_five_candidate_states_match_frozen_thresholds_and_hysteresis() -> None:
    """五条规则必须采用固定尾部阈值，缺失期从满仓开始。"""

    features = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-05", periods=6, freq="7D"),
            "broad_flow_rank_1w": [np.nan, 0.10, 0.50, 0.90, 0.20, 0.10],
            "broad_flow_rank_4w": [np.nan, 0.10, 0.50, 0.90, 0.20, 0.10],
            "hs300_flow_rank_1w": [np.nan, 0.10, 0.50, 0.90, 0.90, 0.10],
        }
    )
    states = build_candidate_weekly_states(features)

    assert states["BROAD_FLOW1W_BOTTOM10_CASH_WEEK"].tolist() == [1, 0, 1, 1, 1, 0]
    assert states["BROAD_FLOW4W_BOTTOM10_CASH_WEEK"].tolist() == [1, 0, 1, 1, 1, 0]
    assert states["BROAD_FLOW1W_DUALTAIL_HYSTERESIS"].tolist() == [1, 0, 0, 1, 1, 0]
    assert states["HS300_FLOW1W_DUALTAIL_HYSTERESIS"].tolist() == [1, 0, 0, 1, 1, 0]
    assert states["BROAD_HS300_DUAL_CONFIRM_20_80_HYSTERESIS"].tolist() == [1, 0, 0, 1, 1, 0]
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8
        assert set(states[candidate_id].unique()).issubset({0, 1})


def test_weekly_feature_can_only_execute_at_t_plus_2_open() -> None:
    """T日周度份额在T+1收盘进入目标序列，因此只能在T+2开盘执行。"""

    market = pd.DataFrame(
        {"trade_date": pd.bdate_range("2024-01-02", periods=6)}
    )
    weekly = pd.DataFrame({"date": [pd.Timestamp("2024-01-03")]})
    for candidate_id in EXPECTED_CANDIDATES:
        weekly[candidate_id] = np.int8(0)
    daily = align_weekly_states_to_daily_targets(market, weekly)

    availability = daily.loc[daily["weekly_feature_became_available"]].iloc[0]
    assert availability["source_feature_date"] == pd.Timestamp("2024-01-03")
    assert availability["trade_date"] == pd.Timestamp("2024-01-04")
    assert daily.loc[daily["trade_date"].eq(pd.Timestamp("2024-01-03")), EXPECTED_CANDIDATES].iloc[0].tolist() == [1] * 5
    assert daily.loc[daily["trade_date"].eq(pd.Timestamp("2024-01-04")), EXPECTED_CANDIDATES].iloc[0].tolist() == [0] * 5
    execution_date = daily.loc[
        daily["trade_date"].gt(availability["trade_date"]), "trade_date"
    ].iloc[0]
    assert execution_date == pd.Timestamp("2024-01-05")
