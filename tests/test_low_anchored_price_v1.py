"""检验新低起点的因果性、量价单位、起点锁定及真实下一开盘退出。"""
import numpy as np
import pandas as pd
import pytest

from research.additional_cycle_exit_account_v1 import simulate_additional_exit
from research.low_anchored_price_v1 import WeightedPriceInputs, build_factors, LockedAnchorExitController


def fixture(values):
    price = np.array(values, float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(price)), "wealth": price, "close": price,
                         "high": price, "low": price, "volume": 100., "feature_valid": True})


def test_new_low_requires_complete_window_and_strictly_lower_price():
    data = fixture([10, 9, 8, 8, 9, 7])
    factors, _ = build_factors(data, 3)
    assert factors.new_anchor.to_list() == [False, False, True, False, False, True]
    assert factors.loc[2, "anchor_index"] == 2 and factors.loc[3, "anchor_index"] == 2
    assert factors.loc[5, "anchor_index"] == 5
    assert factors.loc[[2, 5], "raw_entry"].eq(0).all()


def test_entry_requires_upward_cross_under_the_same_anchor():
    factors, _ = build_factors(fixture([10, 9, 8, 9, 9.2, 8.4, 8.5]), 3)
    assert factors.loc[3, "raw_entry"] == 1
    assert factors.loc[4, "raw_entry"] == 0
    assert factors.loc[5, "new_anchor"] and factors.loc[5, "raw_entry"] == 0
    assert factors.loc[6, "raw_entry"] == 1


def test_future_new_lows_cannot_change_old_anchor_or_signal():
    data = fixture([10, 9, 8, 9, 10, 11, 12, 4, 3, 2])
    full, _ = build_factors(data, 3)
    prefix, _ = build_factors(data.iloc[:7].copy(), 3)
    pd.testing.assert_frame_equal(full.iloc[:7].reset_index(drop=True), prefix, check_dtype=False)


def test_volume_unit_conversion_preserves_weighted_price_and_signals():
    data = fixture([10, 9, 8, 9, 9.2, 8.4, 8.5])
    data["volume"] = [100, 200, 300, 400, 500, 600, 700]
    original, _ = build_factors(data, 3)
    data["volume"] *= 100
    converted, _ = build_factors(data, 3)
    np.testing.assert_allclose(original.weighted_price, converted.weighted_price, equal_nan=True)
    assert original.raw_entry.equals(converted.raw_entry)


def test_raw_price_units_and_dividend_drop_do_not_change_wealth_price_proxy():
    data = fixture([10, 10, 10, 10])
    data.loc[2:, ["high", "low", "close"]] = 9.8
    inputs = WeightedPriceInputs(data)
    assert inputs.average(0, 3)["weighted_price"] == pytest.approx(10.)
    data[["high", "low", "close"]] *= 100
    assert WeightedPriceInputs(data).average(0, 3)["weighted_price"] == pytest.approx(10.)


@pytest.mark.parametrize("bad", ["volume", "close"])
def test_missing_range_is_no_view_and_not_silently_skipped(bad):
    data = fixture([10, 9, 8, 9, 10, 11])
    data.loc[3, bad] = np.nan
    factors, _ = build_factors(data, 3)
    assert factors.loc[4:, "weighted_price"].isna().all()
    assert factors.loc[4:, "raw_entry"].eq(0).all()
    assert factors.loc[4:, "range_status"].str.startswith("NO_VIEW").all()


def test_zero_cumulative_volume_has_no_weighted_price():
    data = fixture([10, 9, 8, 9])
    data["volume"] = 0.
    factors, _ = build_factors(data, 3)
    assert factors.weighted_price.isna().all() and factors.raw_entry.eq(0).all()


def test_locked_entry_anchor_does_not_follow_new_market_low():
    data = fixture([10, 9, 8, 9, 9.2, 8.4, 8.5])
    factors, inputs = build_factors(data, 3)
    controller = LockedAnchorExitController(data, inputs, factors)
    cycle = {"cycle_id": 1, "entry_index": 4}
    assert controller(4, cycle, 100, 100)["locked_anchor_index"] == 2
    later = controller(5, cycle, 100, 100)
    assert later["locked_anchor_index"] == 2 and later["current_market_anchor_index"] == 5
    assert later["locked_weighted_price"] == pytest.approx(np.mean([8, 9, 9.2, 8.4]))
    assert later["below_locked_price"] and later["additional_exit_requested"]


def test_account_buys_next_open_and_blocked_locked_anchor_exit_persists():
    data = fixture([10, 9.5, 9, 9.4, 9.5, 9.2, 9.3, 9.3, 9.4, 9.5])
    data["open"] = data.close.shift(1).fillna(data.close.iloc[0])
    data["previous_close"], data["dividend"] = data.open, 0.
    data.loc[6, "open"] = 8.28
    data["high"] = data[["open", "close"]].max(axis=1)
    data["low"] = data[["open", "close"]].min(axis=1)
    factors, inputs = build_factors(data, 3)
    assert factors.raw_entry.iloc[3] == 1
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": factors.raw_entry.to_numpy(int), "exit": {1: np.zeros(len(data), bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    ledger, decisions, cycles = simulate_additional_exit(data, div, cfg, cost, str(data.date.iloc[1].date()), rule, spec,
                                                       LockedAnchorExitController(data, inputs, factors))
    assert cycles.iloc[0].entry_date == data.date.iloc[4]
    assert ledger.loc[ledger.date.eq(data.date.iloc[6]), "status"].iloc[0] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert cycles.iloc[0].exit_date == data.date.iloc[7]
    assert "固定起点" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-7
