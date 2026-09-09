"""美国杠杆多资产V1采集器的纯合成数据测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.download_us_leveraged_multi_asset_history_v1 import (
    AcquisitionContractError,
    CONFIG,
    load_contract,
    normalize_benchmark,
    normalize_boc_usd_cny,
    normalize_security_history,
    partition_inputs,
)


def _raw_three_days() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "open": [100.0, 50.0, 49.0],
            "high": [101.0, 51.0, 50.0],
            "low": [99.0, 49.0, 48.0],
            "close": [100.0, 50.0, 49.0],
            "volume": [1000.0, 2000.0, 2200.0],
        }
    )


def _factor_split_then_distribution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["1970-01-01", "2020-01-02", "2020-01-03"]),
            "qfq_factor": [0.5, 1.0, 1.0],
            "adjust": [-1.0, -1.0, 0.0],
        }
    )


def test_contract_freezes_user_goal_and_disables_strategy_computation_in_acquisition() -> None:
    contract = load_contract(CONFIG)
    assert contract["objective"]["initial_capital_cny"] == 500000.0
    assert contract["objective"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["objective"]["minimum_annualized_net_excess"] == 0.40
    assert contract["objective"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_tickers"] == ["UPRO", "TQQQ", "TMF", "UGL"]
    assert not contract["data_sources"][
        "strategy_total_return_or_rank_may_be_computed_during_acquisition"
    ]


def test_security_normalization_preserves_split_and_distribution_economics() -> None:
    panel = normalize_security_history(
        "TEST",
        _raw_three_days(),
        _factor_split_then_distribution(),
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-03"),
    )
    assert panel["split_ratio_at_open"].tolist() == pytest.approx([1.0, 2.0, 1.0])
    assert panel["cash_distribution_per_post_event_share_usd"].tolist() == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert panel["signal_total_return_index"].tolist() == pytest.approx(
        [100.0, 100.0, 100.0]
    )
    assert panel["raw_dollar_turnover_usd"].tolist() == pytest.approx(
        [100000.0, 100000.0, 107800.0]
    )


def test_security_normalization_accepts_supplier_datetime_unit_mismatch() -> None:
    raw = _raw_three_days()
    factors = _factor_split_then_distribution()
    raw["date"] = raw["date"].astype("datetime64[s]")
    factors["date"] = factors["date"].astype("datetime64[us]")
    panel = normalize_security_history(
        "TEST",
        raw,
        factors,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-03"),
    )
    assert str(panel["date"].dtype) == "datetime64[ns]"
    assert str(panel["factor_effective_date"].dtype) == "datetime64[ns]"


def test_negative_implied_distribution_is_rejected() -> None:
    factors = _factor_split_then_distribution()
    factors.loc[factors["date"].eq(pd.Timestamp("2020-01-03")), "adjust"] = -2.0
    with pytest.raises(AcquisitionContractError, match="现金分配推导为负"):
        normalize_security_history(
            "TEST",
            _raw_three_days(),
            factors,
            start=pd.Timestamp("2020-01-01"),
            end=pd.Timestamp("2020-01-03"),
        )


def test_boc_normalization_discards_missing_and_converts_per_hundred_quote() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2019-12-30", "2019-12-31", "2020-01-01", "2020-01-02"],
            "中行汇买价": [696.0, 696.0, 696.0, 695.0],
            "中行钞买价": [690.0, 690.0, 690.0, 689.0],
            "中行卖价": [699.0, 699.0, 699.0, 698.0],
            "央行中间价": [698.0, 698.0, 698.0, 697.0],
            "中行折算价": ["698.0", ".", "0", "697.0"],
        }
    )
    result = normalize_boc_usd_cny(raw)
    assert result["date"].tolist() == [pd.Timestamp("2019-12-30"), pd.Timestamp("2020-01-02")]
    assert result["cny_per_usd"].tolist() == pytest.approx([6.98, 6.97])


def test_benchmark_normalization_keeps_positive_unique_total_return_levels() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2019-12-31", "2019-12-31", "2020-01-02"],
            "close": [5000.0, 5001.0, 0.0],
        }
    )
    result = normalize_benchmark(raw)
    assert len(result) == 1
    assert result.iloc[0]["close"] == 5001.0
    assert result.iloc[0]["symbol"] == "H00300"


def test_partition_inputs_is_date_disjoint_and_keeps_sealed_rows_separate() -> None:
    contract = load_contract(CONFIG)
    panel = pd.DataFrame(
        {
            "ticker": ["UPRO", "UPRO", "UPRO"],
            "date": pd.to_datetime(["2019-12-31", "2020-01-02", "2026-08-14"]),
        }
    )
    fx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-30", "2020-01-02", "2026-08-14"]),
            "cny_per_usd": [6.98, 6.97, 7.10],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-31", "2020-01-02", "2026-08-14"]),
            "close": [5000.0, 5010.0, 9000.0],
        }
    )
    result = partition_inputs(panel, fx, benchmark, contract)
    assert result["visible_panel"]["date"].max() == pd.Timestamp("2019-12-31")
    assert result["sealed_panel"]["date"].min() == pd.Timestamp("2020-01-02")
    assert set(result["visible_panel"]["date"]).isdisjoint(result["sealed_panel"]["date"])
