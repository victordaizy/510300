"""核对严格收复、分红边界、事件延续和下一开盘成交。"""
import numpy as np
import pandas as pd
import pytest

from research.gap_recovery_inputs_v1 import recovery_frame
from research.event_clock_account_v1 import simulate_event_account


def prices(opens, closes, cash=None):
    op, cl = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(cl)), "open": op, "close": cl,
        "high": np.maximum(op, cl), "low": np.minimum(op, cl), "previous_close": np.r_[np.nan, cl[:-1]],
        "dividend": np.zeros(len(cl)) if cash is None else cash, "variance60": .0001})


def test_hand_calculated_recovery_holding_exit_and_reentry():
    data = prices([10, 9.8, 10.1, 10.5, 10.25, 10.6, 10.5], [10, 10.2, 10.4, 10.3, 10.5, 10.7, 10.5])
    result = recovery_frame(data)
    np.testing.assert_allclose(result.target, [np.nan, 1, 1, 0, 1, 1, 0], equal_nan=True)
    assert result.entry_event.iloc[5] == 0 and result.target.iloc[5] == 1
    assert result.weak_exit_event.iloc[6] == 1


def test_dividend_only_opening_gap_does_not_create_entry():
    result = recovery_frame(prices([10, 9.3], [10, 9.4], [0, .7]))
    assert result.opening_gap_after_dividend.iloc[1] == 0
    assert result.entry_event.iloc[1] == 0 and result.target.iloc[1] == 0


def test_quote_units_prevent_float_error_from_turning_equal_recovery_into_entry():
    data = prices([.3, .099], [.3, .1], [0, .2])
    assert data.close.iloc[1] + data.dividend.iloc[1] > data.previous_close.iloc[1]
    result = recovery_frame(data)
    assert result.close_recovery_after_dividend.iloc[1] == 0
    assert result.target.iloc[1] == 0


def test_missing_observation_preserves_latent_intent_and_recovers_without_new_entry():
    data = prices([10, 9.8, np.nan, 10.5, 10.7], [10, 10.2, 10.4, 10.6, 10.5])
    result = recovery_frame(data)
    np.testing.assert_allclose(result.target, [np.nan, 1, np.nan, 1, 0], equal_nan=True)
    assert result.policy_intent.iloc[2] == 1 and result.entry_event.iloc[3] == 0


def test_future_prices_do_not_change_prior_intent_or_events():
    data = prices([10, 9.8, 10.1, 10.5, 10.25, 10.6], [10, 10.2, 10.4, 10.3, 10.5, 10.7])
    expected = recovery_frame(data).iloc[:4]
    data.loc[4:, ["open", "close", "previous_close"]] *= 2
    pd.testing.assert_frame_equal(expected, recovery_frame(data).iloc[:4])


@pytest.mark.parametrize("column,value", [("open", -1.), ("close", np.inf), ("dividend", -.01), ("open", 10.0001)])
def test_invalid_input_or_unrepresentable_quote_is_not_repaired(column, value):
    data = prices([10, 9.8], [10, 10.2])
    data.loc[1, column] = value
    with pytest.raises(ValueError):
        recovery_frame(data)


def test_actual_fills_follow_signal_by_one_open_and_terminal_closes_position():
    data = prices([10, 9.8, 10.1, 10.5, 10.25, 10.4, 10.5, 10.7], [10, 10.2, 10.4, 10.3, 10.5, 10.6, 10.7, 10.7])
    target = recovery_frame(data).target.to_numpy()
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(data.date.iloc[1].date()), "收复测试", targets=target, event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[2], data.date.iloc[5]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[4], data.date.iloc[7]]
    assert decisions.iloc[0].signal_state == "NO_VIEW_KEEP_EXISTING_SHARES"
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
