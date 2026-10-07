"""验证会改变研究结论的边界：时点、盘口、聚类、T+1、成交选择与费用。"""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from research.pressure_recovery_v1 import (
    book_features, clustered_expectancy_interval, commission, concession_interval,
    detect_m2_events, expectancy, m1_measure, next_exit_request,
    point_in_time_errors, settle_completed_trade,
)


def close_row():
    return {
        "trade_date": "2026-09-30", "price": 4.0, "reference_low": 4.01, "reference_high": 4.03,
        "price_economic_at": "2026-09-30T15:00:00+08:00",
        "reference_economic_at": "2026-09-30T15:00:00+08:00",
        "received_at": "2026-09-30T15:05:30+08:00",
        "reference_received_at": "2026-09-30T15:05:30+08:00",
        "clock_error_bound_seconds": .1, "close_components_confirmed": True,
        "reference_error_bound_evidence": "仅合成测试", "raw_sha256": "a" * 64, "reference_raw_sha256": "b" * 64,
    }


def test_concession_uses_lower_reference_for_conservative_bound():
    result = concession_interval(4.0, 3.99, 4.02)
    assert result["concession_low_bps"] < 0 < result["concession_high_bps"]


def test_depth_cannot_extrapolate_missing_sell_quantity():
    result = book_features([[3.999, 100], [3.990, 500]], [[4.001, 50], [4.010, 50]], 200)
    assert result["buy_vwap"] is None
    assert result["quoted_available_quantity"] == 100
    assert result["actual_fill_established"] is False
    assert result["bid_visible_depth_cny"] == pytest.approx(399.9)


def test_shallow_visible_book_does_not_prove_full_band_depth():
    result = book_features([[3.999, 100]], [[4.001, 100]], 100)
    assert not result["fixed_band_fully_visible"]
    assert result["buy_vwap"] == 4.001


def test_crossed_book_rejected():
    with pytest.raises(ValueError, match="交叉"):
        book_features([[4.002, 100]], [[4.001, 100]], 100)


def test_late_nav_and_clock_margin_rejected():
    row = close_row()
    row["reference_received_at"] = "2026-09-30T15:05:59.950+08:00"
    assert "LATE_INPUT" in m1_measure(row)["reasons"]


def test_envelope_time_without_economic_reference_rejected():
    row = close_row()
    del row["reference_economic_at"]
    assert m1_measure(row)["status"] == "NO_VIEW"


def test_close_measurement_does_not_fabricate_fill():
    result = m1_measure(close_row())
    assert result["status"] == "MEASUREMENT_ONLY"
    assert result["actual_order"] is False
    assert result["fill_status"].startswith("UNKNOWN")


def test_pre_regime_cannot_use_after_hours_channel():
    row = {k: v.replace("2026-09-30", "2026-07-03") if isinstance(v, str) else v for k, v in close_row().items()}
    assert "BEFORE_AFTER_HOURS_ETF_REGIME" in m1_measure(row)["reasons"]


def test_future_economic_price_rejected():
    row = close_row()
    row["price_economic_at"] = "2026-09-30T15:07:00+08:00"
    assert "FUTURE_ECONOMIC_TIME" in point_in_time_errors(row, "2026-09-30T15:06:00+08:00", True)


def intraday_rows(day, shock=False, recovered=True):
    start = datetime.fromisoformat(day + "T10:00:00+08:00")
    rows = []
    for minute in range(31):
        t = start + timedelta(minutes=minute)
        pressure = shock and 5 <= minute <= 9
        failed = shock and not recovered and minute >= 10
        mid = 3.92 if pressure or failed else 4.0
        ref = 3.96 if shock and minute >= 5 else 4.0
        row = close_row()
        row.update(trade_date=day, decision_at=t.isoformat(), mid=mid, price=mid,
                   reference_low=ref - .0001, reference_high=ref + .0001, reference_mid=ref,
                   price_economic_at=(t-timedelta(seconds=1)).isoformat(),
                   reference_economic_at=(t-timedelta(seconds=1)).isoformat(),
                   received_at=(t-timedelta(seconds=.5)).isoformat(),
                   reference_received_at=(t-timedelta(seconds=.5)).isoformat(),
                   spread_bps=20 if pressure or failed else 5,
                   bid_visible_depth_cny=100 if pressure or failed else 1000,
                   fixed_band_fully_visible=True)
        rows.append(row)
    return rows


def test_m2_fixed_observation_and_same_shock_deduplication():
    history = intraday_rows("2026-08-03") + intraday_rows("2026-08-04")
    rows = history + intraday_rows("2026-08-05", True)
    events = detect_m2_events(rows, baseline_days=2)
    assert len(events) == 1
    assert events[0]["shock_at"] == "2026-08-05T10:05:00+08:00"
    assert events[0]["observation_end_at"] == "2026-08-05T10:10:00+08:00"
    assert events[0]["state"] == "RECOVERED_MEASUREMENT_ONLY"
    assert not events[0]["actual_order"]


def test_unrecovered_event_is_kept():
    rows = intraday_rows("2026-08-03") + intraday_rows("2026-08-04") + intraday_rows("2026-08-05", True, False)
    events = detect_m2_events(rows, baseline_days=2)
    assert len(events) == 1
    assert events[0]["state"] == "NOT_RECOVERED"


def test_missing_endpoint_is_not_postponed_to_success():
    rows = intraday_rows("2026-08-03") + intraday_rows("2026-08-04") + intraday_rows("2026-08-05", True)
    rows = [r for r in rows if r["decision_at"] != "2026-08-05T10:10:00+08:00"]
    assert detect_m2_events(rows, baseline_days=2)[0]["state"] == "NO_VIEW_NO_FIXED_ENDPOINT"


def test_future_days_do_not_supply_baseline():
    rows = intraday_rows("2026-08-03", True) + intraday_rows("2026-08-04") + intraday_rows("2026-08-05")
    assert detect_m2_events(rows, baseline_days=2) == []


def test_holiday_t1_uses_exchange_calendar():
    days = ["2026-09-29", "2026-09-30", "2026-10-08"]
    assert next_exit_request("2026-09-30", days) == "2026-10-08T09:35:00+08:00"
    with pytest.raises(ValueError):
        next_exit_request("2026-10-08", days)


def filled_trade():
    return {"evidence_type": "ACTUAL_BROKER_RECEIPTS", "evidence_path": "仅合成测试",
            "buy_at": "2026-09-30T15:07:00+08:00", "sell_at": "2026-10-08T09:35:01+08:00",
            "exit_request_at": "2026-10-08T09:35:00+08:00", "buy_quantity": 5000, "sell_quantity": 5000,
            "buy_vwap": 4.0, "sell_vwap": 4.01, "buy_commission": 5, "sell_commission": 5,
            "other_fees": 0, "dividend_cash": 0}


def test_actual_prices_do_not_double_charge_spread():
    result = settle_completed_trade(filled_trade(), ["2026-09-30", "2026-10-08"])
    assert result["net_pnl_cny"] == pytest.approx(40)
    assert result["net_return"] == pytest.approx(.002)


@pytest.mark.parametrize("field,value", [("evidence_type", "UNKNOWN"), ("sell_quantity", 4000), ("sell_at", "2026-09-30T15:20:00+08:00")])
def test_unknown_partial_and_same_day_exit_not_settled(field, value):
    trade = filled_trade()
    trade[field] = value
    with pytest.raises(ValueError):
        settle_completed_trade(trade, ["2026-09-30", "2026-10-08"])


def test_commission_minimum_by_executed_order_and_no_fill_no_fee():
    assert commission(0) == 0
    assert 2 * commission(5000) / 5000 * 1e4 == 20
    assert 2 * commission(20000) / 20000 * 1e4 == 5
    assert 2 * commission(200000) / 200000 * 1e4 == 4
    assert commission(10000) + commission(10000) > commission(20000)


def test_flats_are_in_denominator_and_tail_can_reverse_expectancy():
    result = expectancy([{"net_return": r} for r in [.012] * 60 + [-.008] * 35 + [-.08] * 5])
    assert result["win_rate"] == .6
    assert result["net_expectancy"] - .0014 == pytest.approx(-.001)
    flat = expectancy([{"net_return": r} for r in [.02, -.01, 0]])
    assert flat["flat_rate"] == pytest.approx(1/3)
    assert flat["net_expectancy"] == pytest.approx(.01/3)


def test_no_trade_means_unknown_not_zero_expectancy():
    assert expectancy([])["net_expectancy"] is None
    rows = [{"trade_date": "2026-09-30", "net_return": .01} for _ in range(500)]
    result = clustered_expectancy_interval(rows)
    assert result["independent_days"] == 1
    assert result["lower"] is None
