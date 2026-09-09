"""核对严格波动边界、缺失和实际下一开盘切换。"""
import numpy as np
import pandas as pd
import pytest

from research.event_clock_account_v1 import simulate_event_account
from research.volatility_expert_router_inputs_v1 import routing_frame


def fixture(short=None, panic=None, learned=None):
    n = 8
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "variance60": .0001,
        "vol5": np.ones(n) if short is None else short, "vol60": 1.})
    a = np.ones(n) if panic is None else np.asarray(panic, float)
    b = np.ones(n) if learned is None else np.asarray(learned, float)
    states = pd.DataFrame({"date": data.date, "panic_state": a, "learned_state": b, "target": .5 * (a + b)})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    return data, states, div, cfg, cost


def account(args):
    data, states, div, cfg, cost = args
    return simulate_event_account(data, div, cfg, cost, str(data.date.iloc[1].date()), "波动切换测试",
        targets=routing_frame(data, states).routed_target.to_numpy(), event_mask=np.ones(len(data), bool))


def test_exact_boundary_and_large_short_vol_choose_different_experts():
    data, states, *_ = fixture(short=[1.5, 1.50001, 0, 1, 1, 1, 1, 1], panic=np.ones(8), learned=np.zeros(8))
    result = routing_frame(data, states)
    assert result.routed_target.iloc[:3].tolist() == [0., 1., 0.]
    assert result.selected_expert.iloc[:3].tolist() == ["LEARNED", "PANIC", "LEARNED"]


def test_unknown_selected_expert_and_zero_long_vol_stay_no_view():
    data, states, *_ = fixture(short=[2, 1, 1, 1, 1, 1, 1, 1])
    states.loc[0, "panic_state"] = np.nan
    data.loc[1, "vol60"] = 0
    data.loc[2, "vol5"] = np.nan
    assert routing_frame(data, states).routed_target.iloc[:3].isna().all()


def test_unselected_missing_expert_does_not_invalidate_known_selection():
    data, states, *_ = fixture()
    states.loc[0, "panic_state"] = np.nan
    assert routing_frame(data, states).routed_target.iloc[0] == 1


@pytest.mark.parametrize("column,value", [("vol5", -1.), ("vol60", np.inf)])
def test_invalid_volatility_is_rejected(column, value):
    data, states, *_ = fixture()
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="波动率"):
        routing_frame(data, states)


def test_invalid_state_is_rejected_even_if_not_selected():
    data, states, *_ = fixture()
    states.loc[0, "panic_state"] = 2
    with pytest.raises(ValueError, match="专家状态"):
        routing_frame(data, states)


def test_future_changes_cannot_change_prior_choice():
    data, states, *_ = fixture()
    expected = routing_frame(data, states).iloc[:4]
    data.loc[4:, "vol5"] = 2
    states.loc[4:, "panic_state"] = 0
    pd.testing.assert_frame_equal(routing_frame(data, states).iloc[:4], expected)


def test_switching_expert_with_same_target_does_not_round_trip():
    ledger, decisions = account(fixture(short=[1, 2, 1, 2, 1, 2, 1, 1]))
    assert ledger.iloc[0].filled_quantity > 0
    assert ledger.iloc[1:-1].filled_quantity.eq(0).all()
    assert ledger.iloc[-1].shares == 0
    assert decisions.reference_weight.eq(1).all()


def test_high_vol_switch_to_zero_exits_at_next_open_then_can_reenter():
    args = fixture(short=[1, 2, 1, 1, 1, 1, 1, 1], panic=np.zeros(8))
    ledger, decisions = account(args)
    assert ledger.iloc[0].shares > 0
    assert ledger.iloc[1].filled_quantity < 0 and ledger.iloc[1].shares == 0
    assert ledger.iloc[2].filled_quantity > 0
    assert decisions.iloc[1].execution_date == args[0].date.iloc[2]
    assert ledger.accounting_error.abs().max() < 1e-7


def test_missing_regime_preserves_shares_and_unknown_target():
    args = fixture(short=[1, np.nan, 2, 1, 1, 1, 1, 1], panic=np.zeros(8))
    ledger, decisions = account(args)
    assert ledger.iloc[1].shares > 0 and ledger.iloc[1].filled_quantity == 0
    assert pd.isna(decisions.iloc[1].reference_weight)
    assert decisions.iloc[1].signal_state == "NO_VIEW_KEEP_EXISTING_SHARES"
    assert ledger.iloc[2].shares == 0
