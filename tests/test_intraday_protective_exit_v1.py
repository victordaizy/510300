"""核对新增成交时点、分红、T+1和退出受阻状态，不搜索历史参数。"""
import numpy as np
import pandas as pd
import pytest

from research.intraday_protective_exit_v1 import simulate_protective
from research.simple_price_entry_exit_v1 import simulate_policy


def fixture_data():
    dates = pd.bdate_range("2020-01-01", periods=7)
    frame = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.1,
                          "low": 9.9, "previous_close": 10., "dividend": 0.})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.array([1, 0, 0, 0, 0, 0, 0]), "exit": {1: np.zeros(7, dtype=bool)}}
    spec = {"cooldown": 1, "modes": {1: {"loss": .05, "trail": .05, "take": None, "days": None}}}
    return frame, div, cfg, cost, str(dates[1].date()), rule, spec


def test_disabled_matches_existing_account():
    args = fixture_data()
    old, old_decisions, old_cycles = simulate_policy(*args)
    new, decisions, cycles = simulate_protective(*args, assumption="DISABLED")
    for column in ["equity", "cash", "shares", "net_return", "dividend_receivable", "commission", "slippage_cost"]:
        np.testing.assert_allclose(new[column], old[column], rtol=0, atol=1e-9)
    pd.testing.assert_frame_equal(decisions, old_decisions)
    pd.testing.assert_frame_equal(cycles, old_cycles)


def test_new_purchase_cannot_exit_same_day_then_old_position_can():
    args = fixture_data()
    args[0].loc[[1, 2], "low"] = 9.4
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    assert ledger.iloc[0].filled_quantity == 10000
    assert not ledger.iloc[0].conditional_intraday_fill
    assert ledger.iloc[1].filled_quantity == -10000
    assert ledger.iloc[1].open_price == pytest.approx(9.5)
    assert cycles.iloc[0].holding_intervals == 1
    assert ledger.accounting_error.abs().max() < 1e-7


def test_gap_through_protection_uses_open_not_stop():
    args = fixture_data()
    args[0].loc[2, ["open", "low", "close"]] = [9.4, 9.3, 9.5]
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    row = ledger.iloc[1]
    assert row.execution_clock == "OPEN_PROTECTIVE"
    assert row.open_price == pytest.approx(9.4)
    assert row.protective_price == pytest.approx(9.5)


def test_current_high_does_not_raise_today_stop():
    args = fixture_data()
    args[0].loc[2, ["high", "low", "close"]] = [11., 9.7, 10.]
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    assert ledger.iloc[1].protective_price == pytest.approx(9.5)
    assert ledger.iloc[1].filled_quantity == 0


def test_limit_blocked_exit_latches_until_next_open():
    args = fixture_data()
    args[6]["modes"][1].update(loss=.1, trail=None)
    args[0].loc[2, "low"] = 9.
    args[0].loc[3, "open"] = 10.1
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    assert ledger.iloc[1].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.iloc[1].shares == 10000
    assert ledger.iloc[2].filled_quantity == -10000
    assert ledger.iloc[2].open_price == pytest.approx(10.1)
    assert "盘中" in cycles.iloc[0].exit_reasons


def test_dividend_preserves_value_and_adjusts_protection():
    args = list(fixture_data())
    dates = args[0].date
    args[1] = pd.DataFrame([{"record_date": dates[1], "ex_date": dates[2], "payment_date": dates[3],
                             "cash_dividend_per_share": .2}])
    args[0].loc[2:, ["open", "close"]] = 9.8
    args[0].loc[2:, "low"] = 9.4
    args[0].loc[3:, "previous_close"] = 9.8
    args[0].loc[2, "dividend"] = .2
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    assert ledger.iloc[1].protective_price == pytest.approx(9.3)
    assert ledger.iloc[1].filled_quantity == 0
    assert ledger.iloc[1].equity == pytest.approx(100000)
    assert ledger.iloc[1].dividend_receivable == pytest.approx(2000)
    assert ledger.iloc[2].dividend_paid == pytest.approx(2000)
    assert ledger.accounting_error.abs().max() < 1e-7


def test_adverse_fill_changes_price_not_trigger():
    args = fixture_data()
    args[0].loc[2, "low"] = 9.4
    normal, _, cycles_normal = simulate_protective(*args, assumption="TRIGGER")
    adverse, _, cycles_adverse = simulate_protective(*args, assumption="DAY_LOW")
    assert normal.iloc[1].protective_price == adverse.iloc[1].protective_price
    assert normal.iloc[1].open_price == pytest.approx(9.5)
    assert adverse.iloc[1].open_price == pytest.approx(9.4)
    assert normal.iloc[-1].equity - adverse.iloc[-1].equity == pytest.approx(1000)


def test_terminal_never_uses_intraday_low():
    args = fixture_data()
    args[0].loc[6, "low"] = 9.4
    ledger, decisions, cycles = simulate_protective(*args, assumption="TRIGGER")
    assert ledger.iloc[-1].execution_clock == "OPEN_TERMINAL"
    assert ledger.iloc[-1].open_price == pytest.approx(10)
    assert not ledger.iloc[-1].conditional_intraday_fill
