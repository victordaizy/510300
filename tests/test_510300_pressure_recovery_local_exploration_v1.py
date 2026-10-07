"""防止本地探索偷看未来、删掉失败观察或把价格假设变成真实成交。"""

import pandas as pd

from research.pressure_recovery_local_exploration_v1 import (
    detect_price_episodes, hypothetical_price_cost, next_day_proxies,
    quote_at, visible_order_quote,
)


def example_grid():
    times = pd.date_range("2026-07-09 09:35", periods=23, freq="min", tz="Asia/Shanghai")
    returns = [0.0] * len(times)
    for index in (0, 3, 16):
        returns[index] = -6.0
    return pd.DataFrame({"decision_at": times, "log_mid_return_5m_bps": returns,
                         "mid": [4.0] * len(times), "spread_ticks": [1.0] * len(times),
                         "bid_depth_cny": [100000.0] * len(times)})


def test_measurement_never_uses_future_quote_but_entry_is_strictly_after_observation():
    quotes = pd.DataFrame({"_at": pd.to_datetime(["2026-07-09 09:30:01+08:00", "2026-07-09 09:30:04+08:00"])})
    target = pd.Timestamp("2026-07-09 09:30:00+08:00")
    assert quote_at(quotes, target) is None
    assert quote_at(quotes, target, after=True)._at == target + pd.Timedelta(seconds=1)
    assert quote_at(quotes, target + pd.Timedelta(seconds=3))._at == target + pd.Timedelta(seconds=1)
    assert quote_at(quotes, target + pd.Timedelta(seconds=10)) is None


def test_overlapping_pressure_windows_merge_until_ten_quiet_minutes():
    events = detect_price_episodes(example_grid())
    assert [event["shock_at"].strftime("%H:%M") for event in events] == ["09:35", "09:51"]
    assert events[0]["observation_end_at"].strftime("%H:%M") == "09:40"


def test_nonrebound_and_missing_endpoint_events_are_both_retained():
    grid = example_grid()
    grid.loc[grid.decision_at.dt.strftime("%H:%M").eq("09:40"), "mid"] = 3.9
    events = detect_price_episodes(grid)
    assert events[0]["observation_price_change_bps"] < 0
    grid = grid.loc[~grid.decision_at.dt.strftime("%H:%M").eq("09:40")]
    events = detect_price_episodes(grid)
    assert events[0]["observation_status"] == "NO_FIXED_ENDPOINT"
    assert events[0]["event_id"] == "P0_20260709_0935"


def test_visible_quantity_obeys_lot_and_cash_budget_including_commission():
    row = {}
    for level in range(1, 11):
        row[f"ask_px{level}"] = 40000 + level * 10
        row[f"bid_px{level}"] = 40010 - level * 10
        row[f"ask_vol{level}"] = 100000
        row[f"bid_vol{level}"] = 100000
    result = visible_order_quote(pd.Series(row), 20000)
    assert result["quantity"] % 100 == 0
    assert 0 < result["cash_required_cny"] <= 20000
    assert result["same_snapshot_roundtrip_cost_bps"] > 0
    assert result["next_day_minute_proxy_breakeven_cny"] > result["buy_vwap_cny"]
    assert result["actual_fill_established"] is False


def test_missing_next_day_is_not_replaced_by_later_available_day():
    prices = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-13 09:35"]),
                           "close": [4.1], "high": [4.2], "low": [4.0]})
    result = next_day_proxies("2026-07-09", prices, ["2026-07-09", "2026-07-10", "2026-07-13"])
    assert result["next_trade_date"] == "2026-07-10"
    assert result["prices"] == {}
    assert result["status"] == "NEXT_DAY_MINUTE_DATA_MISSING_OR_PARTIAL"


def test_flat_price_scenario_loses_cost_and_never_becomes_verified_return():
    result = hypothetical_price_cost(4900, 4.0, 4.0)
    assert result["gross_price_path_bps"] == 0
    assert result["hypothetical_cash_difference_cny"] < 0
    assert result["actual_fill_established"] is False
    assert result["verified_strategy_return"] is False
