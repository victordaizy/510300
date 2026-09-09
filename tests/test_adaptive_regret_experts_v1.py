"""合成数据检验比较损失、月度生效时钟及下一开盘实际成交。"""
import math
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from research.adaptive_regret_experts_inputs_v1 import normalized_record_weights, adaptive_regret_frame, prefix_factors
from research.event_clock_account_v1 import simulate_event_account


def fixture():
    dates = pd.bdate_range("2020-01-31", periods=48)
    returns = np.tile([.02, -.01], (len(dates), 1))
    states = np.tile([1., 0.], (len(dates), 1))
    return dates, returns, states


def direct_weight(r, c):
    return (math.exp(max(r+1., 0.)**2/(3.*(c+1.)))-math.exp(max(r-1., 0.)**2/(3.*(c+1.))))/2.


def test_potential_matches_hand_values_and_normalizes():
    r, c = np.array([[.3, -.2, 0.], [.8, -1.2, .1]]), np.array([[.5, .6, 0.], [.9, 1.2, .4]])
    values = np.array([[direct_weight(x, y) for x, y in zip(a, b)] for a, b in zip(r, c)])
    actual, fallback = normalized_record_weights(r, c)
    np.testing.assert_allclose(actual, values/values.sum(), rtol=1e-14, atol=1e-15)
    assert not fallback and actual[1, 1] == 0.


def test_large_potential_stays_finite_without_direct_exponential():
    actual, fallback = normalized_record_weights([[10000., 9999., -10000.]], [[10000., 10000., 10000.]])
    assert np.isfinite(actual).all() and actual[0, 0] > actual[0, 1] > actual[0, 2] == 0.
    assert np.isclose(actual.sum(), 1.) and not fallback


def test_exact_all_zero_weights_have_explicit_equal_record_fallback():
    actual, fallback = normalized_record_weights([[-1., -2., -3.], [-4., -5., -6.]], [[1., 2., 3.], [4., 5., 6.]])
    np.testing.assert_array_equal(actual, np.full((2, 3), 1./6.))
    assert fallback


def test_invalid_regret_bound_is_rejected_without_repair():
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        normalized_record_weights([[.2, .1, 0.]], [[.1, .1, 0.]])


def test_first_update_uses_old_weights_and_birth_has_no_past_losses():
    dates, returns, states = fixture()
    frame, records = adaptive_regret_frame(dates, returns, states, 1)
    loss = np.array([.49, .505, .5])
    expected = np.mean(loss)-loss
    np.testing.assert_allclose(frame.loc[1, ["panic_regret", "learned_regret", "cash_regret"]].to_numpy(float), expected, atol=1e-15)
    a, b = int(frame.record_start.iloc[1]), int(frame.record_stop.iloc[1])
    np.testing.assert_allclose(records["regret"][a:b].reshape(2, 3), np.vstack([expected, np.zeros(3)]), atol=1e-15)
    assert list(records["group_birth_index"][:2]) == [0, 1]
    assert frame.updated_records.iloc[1] == 3 and frame.active_groups.iloc[1] == 2
    original = np.r_[[direct_weight(r, abs(r)) for r in expected], [direct_weight(0., 0.)]*3]
    budget = (original/original.sum()).reshape(2, 3).sum(axis=0)
    assert np.isclose(frame.virtual_loss.iloc[2], budget @ loss)
    assert not np.isclose(frame.virtual_loss.iloc[2], frame.loc[2, ["panic_budget", "learned_budget", "cash_budget"]].to_numpy(float) @ loss, rtol=0., atol=1e-10)


def test_same_zero_returns_are_valid_without_a_variance_gate():
    dates, returns, states = fixture()
    returns[:] = 0.
    states[:] = 1.
    frame, records = adaptive_regret_frame(dates, returns, states, 1)
    np.testing.assert_allclose(frame[["panic_budget", "learned_budget", "cash_budget"]].iloc[:-1], 1./3., atol=1e-14)
    np.testing.assert_allclose(frame.target.iloc[:-1], 2./3., atol=1e-14)
    assert frame.learning_status.iloc[:-1].eq("ADAPTIVE_REGRET_AVAILABLE").all()
    assert np.abs(records["regret"]).max() < 1e-12


def test_future_observations_and_later_births_do_not_change_earlier_weights():
    dates, returns, states = fixture()
    frame, _ = adaptive_regret_frame(dates, returns, states, 1)
    changed_r, changed_s = returns.copy(), states.copy()
    changed_r[20:], changed_s[20:] = [-.2, .3], [0., 1.]
    changed, _ = adaptive_regret_frame(dates, changed_r, changed_s, 1)
    assert_frame_equal(frame.iloc[:20], changed.iloc[:20])
    short, _ = adaptive_regret_frame(dates[:21], returns[:21], states[:21], 1)
    source = pd.DataFrame({"date": dates[:21], "panic_reference_return": returns[:21, 0], "learned_reference_return": returns[:21, 1],
                           "panic_state": states[:21, 0], "learned_state": states[:21, 1]})
    assert_frame_equal(short, prefix_factors(frame, source))
    returns[-1], states[-1] = [100., -5.], [np.nan, np.nan]
    terminal, _ = adaptive_regret_frame(dates, returns, states, 1)
    assert_frame_equal(frame.iloc[:-1], terminal.iloc[:-1])
    assert np.isnan(terminal.target.iloc[-1])


def test_missing_or_out_of_domain_returns_and_intents_stop_remainder():
    dates, returns, states = fixture()
    for changed_return, changed_state in [(np.nan, 1.), (-1., 1.), (1.01, 1.), (.02, np.nan), (.02, .5)]:
        r, s = returns.copy(), states.copy()
        r[4, 0], s[4, 0] = changed_return, changed_state
        frame, _ = adaptive_regret_frame(dates, r, s, 1)
        assert frame.learning_status.iloc[4:-1].eq("NO_VIEW_SOURCE_GAP_REMAINDER").all()
        assert frame.target.iloc[4:].isna().all() and not frame.birth_added.iloc[4:].any()
    returns[4, 0] = 1.
    frame, _ = adaptive_regret_frame(dates, returns, states, 1)
    assert frame.learning_status.iloc[4] == "ADAPTIVE_REGRET_AVAILABLE" and frame.panic_loss.iloc[4] == 0.


def test_numerical_failure_is_no_view_without_equal_cash_substitution(monkeypatch):
    from research import adaptive_regret_experts_inputs_v1 as module
    def broken(*args):
        raise FloatingPointError("合成数值失败")
    monkeypatch.setattr(module, "normalized_record_weights", broken)
    frame, _ = module.adaptive_regret_frame(*fixture(), 1)
    assert frame.learning_status.iloc[:-1].eq("NO_VIEW_NUMERICAL_REMAINDER").all()
    assert frame.target.isna().all() and frame.panic_budget.isna().all()


def test_actual_next_open_dividend_and_unconditional_full_exit():
    dates, returns, states = fixture()
    returns[:] = 0.
    states[:] = [1., 0.]
    states[6:] = 0.
    frame, _ = adaptive_regret_frame(dates, returns, states, 1)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    data.loc[4:, ["open", "close", "previous_close"]] = 9.9
    data.loc[4, ["previous_close", "dividend"]] = [10., .1]
    div = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "合成动态预算", targets=frame.target.to_numpy(), event_mask=np.ones(len(dates), bool))
    assert ledger.filled_quantity.iloc[0] == 6600 and decisions.execution_date.iloc[0] == dates[1]
    assert ledger.loc[ledger.date.eq(dates[6]), "shares"].iloc[0] == 6600
    assert ledger.loc[ledger.date.eq(dates[7]), "shares"].iloc[0] == 0
    assert ledger.dividend_recognized.sum() == 660. and ledger.dividend_paid.sum() == 660.
    assert ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0 and ledger.accounting_error.abs().max() < 1e-6
    assert ledger.shares.iloc[-1] == 0


def test_missing_target_keeps_real_shares_until_terminal_open():
    dates, returns, states = fixture()
    states[4, 0] = np.nan
    frame, _ = adaptive_regret_frame(dates, returns, states, 1)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "variance60": .001})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    ledger, decisions = simulate_event_account(data, div, cfg, cost, str(dates[1].date()), "合成无新观点", targets=frame.target.to_numpy(), event_mask=np.ones(len(dates), bool))
    assert decisions.loc[decisions.origin.ge(dates[4]), "reference_weight"].isna().all()
    assert ledger.shares.iloc[3:-1].eq(6600).all() and ledger.shares.iloc[-1] == 0
