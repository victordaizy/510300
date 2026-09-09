"""美国中国ETF隔夜信息二元筛选V1的时钟、状态与账本回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday_binary_livermore_screen_v1 import CostModel  # noqa: E402
from research.us_china_overnight_binary_screen_v1 import (  # noqa: E402
    CANDIDATE_CONTRACT_SHA256,
    EXPECTED_CANDIDATES,
    build_candidate_states,
    load_candidate_contract,
    sha256_file,
    simulate_preopen_candidate,
    strict_preopen_interval_returns,
)


def _us_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "adj_close": [100.0, 110.0, 121.0, 133.1],
            "symbol": symbol,
            "source": "合成测试",
        }
    )


def _costs() -> CostModel:
    return CostModel(
        scenario="TEST",
        commission_rate=0.0001,
        minimum_commission=0.0,
        slippage_bps=5.0,
        cash_annual_rate=0.015,
        trading_days_per_year=242,
        lot_size=100,
    )


def test_strict_preopen_alignment_excludes_same_calendar_day_and_accumulates_holiday() -> None:
    """D日不得读取美国D日线，中国休市间隔应累计多个已结束美国时段。"""

    china_dates = pd.Series(
        pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"])
    )
    aligned = strict_preopen_interval_returns(china_dates, _us_frame("ASHR"), "ASHR")

    assert np.isnan(aligned.loc[0, "ASHR_interval_return"])
    assert aligned.loc[1, "ASHR_latest_us_date"] == pd.Timestamp("2024-01-03")
    assert aligned.loc[1, "ASHR_baseline_us_date"] == pd.Timestamp("2024-01-02")
    assert np.isclose(aligned.loc[1, "ASHR_interval_return"], 0.10)
    assert aligned.loc[1, "ASHR_new_session_count"] == 1
    assert aligned.loc[2, "ASHR_latest_us_date"] == pd.Timestamp("2024-01-05")
    assert aligned.loc[2, "ASHR_baseline_us_date"] == pd.Timestamp("2024-01-03")
    assert np.isclose(aligned.loc[2, "ASHR_interval_return"], 0.21)
    assert aligned.loc[2, "ASHR_new_session_count"] == 2
    assert bool((aligned["ASHR_latest_us_date"].dropna() < aligned.loc[aligned["ASHR_latest_us_date"].notna(), "date"]).all())


def test_eight_candidate_rules_have_exact_binary_mapping_and_missing_defaults_full() -> None:
    """验证延续、复合风险、广度与过度反应规则及缺失回退。"""

    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "signal_asof_us_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "signal_complete": [True, True, False],
            "ASHR_interval_return": [-0.01, 0.03, -0.05],
            "china_basket_return": [-0.02, 0.02, -0.05],
            "china_specific_return": [-0.03, 0.01, -0.06],
            "all4_negative": [True, False, True],
            "rank_ashr": [0.05, 0.90, 0.01],
            "rank_china_basket": [0.10, 0.90, 0.01],
            "rank_china_specific": [0.10, 0.50, 0.01],
            "rank_vix_return": [0.80, 0.50, 0.99],
        }
    )

    states = build_candidate_states(features)

    assert states["USCN_ASHR_NEG_CONTINUATION_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_BASKET_NEG_CONTINUATION_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_BASKET_BOTTOM10_CONTINUATION_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_SPECIFIC_BOTTOM10_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_ALL4_NEG_BREADTH_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_BASKET_BOTTOM20_VIX_TOP80_CASH"].tolist() == [0, 1, 1]
    assert states["USCN_ASHR_TOP10_OVERREACTION_CASH"].tolist() == [1, 0, 1]
    assert states["USCN_BASKET_TOP10_OVERREACTION_CASH"].tolist() == [1, 0, 1]
    assert list(states.columns[-8:]) == EXPECTED_CANDIDATES
    for candidate_id in EXPECTED_CANDIDATES:
        assert states[candidate_id].dtype == np.int8
        assert set(states[candidate_id].unique()).issubset({0, 1})


def test_preopen_state_executes_at_same_day_open_without_t_plus_one_violation() -> None:
    """D日09:15已知状态应在D日开盘执行，最早于下一交易日卖出。"""

    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "open": [5.00, 5.10, 5.20],
            "close": [5.05, 5.15, 5.25],
        }
    )
    target_states = np.array([0, 1, 0], dtype=np.int8)
    signal_asof_dates = pd.Series(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    dividends = pd.DataFrame(
        columns=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source",
        ]
    )

    ledger, trades, diagnostics = simulate_preopen_candidate(
        market,
        dividends,
        target_states,
        signal_asof_dates,
        start_date=pd.Timestamp("2024-01-04"),
        end_date=pd.Timestamp("2024-01-05"),
        costs=_costs(),
        initial_capital=500000.0,
    )

    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert pd.Timestamp(trades.iloc[0]["date"]) == pd.Timestamp("2024-01-04")
    assert pd.Timestamp(trades.iloc[1]["date"]) == pd.Timestamp("2024-01-05")
    assert diagnostics["t_plus_one_violations"] == 0
    assert diagnostics["same_or_future_signal_date_violations"] == 0
    assert set(ledger["actual_state"].unique()).issubset({0, 1})


def test_candidate_contract_hash_and_double_twenty_gate_are_immutable() -> None:
    """数据下载前候选合同、八规则和双20门必须保持不变。"""

    contract = load_candidate_contract()

    assert sha256_file(ROOT / "config" / "510300_us_china_overnight_binary_screen_v1_candidates.yaml") == CANDIDATE_CONTRACT_SHA256
    assert [item["candidate_id"] for item in contract["candidates"]] == EXPECTED_CANDIDATES
    assert contract["objective"]["minimum_annualized_excess"] == 0.20
    assert contract["objective"]["minimum_rolling_excess_median"] == 0.20
    assert contract["scope"]["allowed_target_states"] == [0, 1]
