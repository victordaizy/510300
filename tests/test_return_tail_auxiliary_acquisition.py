from __future__ import annotations

import pandas as pd

from research.return_tail_auxiliary_acquisition import (
    active_members,
    build_constituent_panel,
    normalize_industry_intervals,
    normalize_stock_history,
)


def test_normalize_stock_history_applies_factor_and_units() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260814"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "vol": [2.0],
            "amount": [3.0],
        }
    )
    factor = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20260814"], "adj_factor": [2.0]}
    )
    result = normalize_stock_history(daily, factor, "000001.SZ", "2026-08-16")
    assert result.loc[0, "total_return_close"] == 21.0
    assert result.loc[0, "volume"] == 200.0
    assert result.loc[0, "amount"] == 3000.0


def test_active_members_uses_left_closed_right_open_intervals() -> None:
    intervals = pd.DataFrame(
        {
            "symbol": [f"{value:06d}.SZ" for value in range(300)]
            + ["999999.SH"],
            "opt_in": ["2020-01-01"] * 300 + ["2020-01-02"],
            "opt_out": [None] * 299 + ["2020-01-02", None],
        }
    )
    result = active_members(intervals, ["2020-01-01", "2020-01-02"])
    day_two = set(result.loc[result["date"].eq(pd.Timestamp("2020-01-02")), "con_code"])
    assert "000299.SZ" not in day_two
    assert "999999.SH" in day_two


def test_constituent_panel_forward_fills_only_suspension_price() -> None:
    symbols = [f"{value:06d}.SZ" for value in range(300)]
    intervals = pd.DataFrame(
        {"symbol": symbols, "opt_in": ["2020-01-01"] * 300, "opt_out": [None] * 300}
    )
    rows = []
    for symbol in symbols:
        for date in ["2019-12-31", "2020-01-01"]:
            if symbol == "000000.SZ" and date == "2020-01-01":
                continue
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "con_code": symbol,
                    "raw_open": 10.0,
                    "raw_high": 10.0,
                    "raw_low": 10.0,
                    "raw_close": 10.0,
                    "total_return_open": 20.0,
                    "total_return_high": 20.0,
                    "total_return_low": 20.0,
                    "total_return_close": 20.0,
                    "adj_factor": 2.0,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "price_source": "测试",
                    "adjustment_source": "测试",
                    "retrieved_at": "2026-08-16",
                }
            )
    panel = build_constituent_panel(
        pd.DataFrame(rows), intervals, ["2019-12-31", "2020-01-01"], ["2020-01-01"]
    )
    suspended = panel.loc[panel["con_code"].eq("000000.SZ")].iloc[0]
    assert bool(suspended["is_suspended"])
    assert suspended["total_return_close"] == 20.0
    assert suspended["volume"] == 0.0


def test_industry_intervals_preserve_dates_and_point_in_time_label() -> None:
    raw = pd.DataFrame(
        {
            "l1_code": ["801010.SI"],
            "l1_name": ["农林牧渔"],
            "l2_code": ["801011.SI"],
            "l2_name": ["种植业"],
            "l3_code": ["850111.SI"],
            "l3_name": ["种子"],
            "ts_code": ["000001.SZ"],
            "in_date": ["20200101"],
            "out_date": ["20211210"],
            "is_new": ["N"],
        }
    )
    result = normalize_industry_intervals(raw, {"000001.SZ"}, "2026-08-16")
    assert result.loc[0, "in_date"] == pd.Timestamp("2020-01-01")
    assert result.loc[0, "out_date"] == pd.Timestamp("2021-12-10")
    assert result.loc[0, "classification_usage"] == "POINT_IN_TIME_INTERVAL"
