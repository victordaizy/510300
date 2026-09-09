"""验证分批不提前看成交、严格确认、重置和下一开盘执行。"""
import numpy as np
import pandas as pd
from research.staged_entry_inputs_v1_1 import reference_entry_frame, staged_targets
from research.event_clock_account_v1 import simulate_event_account
from tests.test_median_continuation_v1 import fixture


def inputs():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=8), "wealth_close": [1.2, 1., 1.01, .9, .9, 1., 1.01, 1.],
        "reference_new_buy": [False, True, False, False, False, True, False, False],
        "reference_holding": [False, True, True, True, False, True, True, False],
        "reference_entry_wealth_open": [np.nan, 1., 1., 1., np.nan, 1., 1., np.nan]})


def test_no_confirmation_before_reference_fill_and_strict_equality():
    result = staged_targets(inputs(), [1, 1, 1, 1, 0, 1, 1, np.nan], 1)
    np.testing.assert_allclose(result.target.iloc[:7], [.5, .5, 1., 1., 0., .5, 1.])
    assert result.price_upgrade.sum() == 2


def test_missing_view_keeps_confirmed_stage_without_filling_target():
    result = staged_targets(inputs(), [1, 1, 1, np.nan, 1, 1, 1, np.nan], 1)
    assert np.isnan(result.target.iloc[3]) and result.stage.iloc[3] == 1.
    assert result.target.iloc[4] == 1.
    assert result.target.iloc[5] == .5


def test_reference_exit_resets_the_stage():
    frame = inputs()
    frame.loc[4, "reference_new_buy"] = False
    result = staged_targets(frame, [1, 1, 1, 0, 1, 1, 1, np.nan], 1)
    assert result.target.iloc[3] == 0 and result.target.iloc[4] == .5


def test_reference_wealth_open_uses_known_distribution_and_no_backward_fill():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5), "wealth": [1., 1.01, 1.02, 1.03, 1.03],
        "open": [10., 9., 9.1, 9.2, 9.3], "dividend": [0., 1., 0., 0., 0.], "previous_close": [np.nan, 10., 9.1, 9.2, 9.3]})
    ledger = pd.DataFrame({"date": data.date.iloc[1:].to_numpy(), "shares_before": [0, 100, 100, 0],
        "filled_quantity": [100, 0, -100, 0], "shares_after": [100, np.nan, 0, np.nan], "shares": [100, 100, 0, 0]})
    f = reference_entry_frame(data, ledger, 1)
    np.testing.assert_allclose(f.reference_entry_wealth_open.to_numpy(), [np.nan, 1., 1., np.nan, np.nan], equal_nan=True)
    assert f.reference_entry_date.iloc[0] is pd.NaT and f.reference_entry_date.iloc[2] == data.date.iloc[1]


def test_future_changes_leave_prior_decisions_identical():
    first = inputs()
    state = np.array([1, 1, 1, 1, 0, 1, 1, np.nan], float)
    result = staged_targets(first, state, 1)
    changed = first.copy()
    changed.loc[5:, "wealth_close"] *= 100
    changed.loc[5:, "reference_new_buy"] = True
    pd.testing.assert_frame_equal(result.iloc[:5], staged_targets(changed, state, 1).iloc[:5])


def test_two_buys_then_next_open_exit_obeys_complete_account_clock():
    data, div, cfg, cost, start, rule, spec = fixture()
    data["variance60"] = .01
    cfg["weight_band"] = .1
    targets = np.array([.5, 1., 0., 0., 0., 0., 0., 0., 0., np.nan])
    ledger, decisions = simulate_event_account(data, div, cfg, cost, start, "STAGED_ENTRY", targets=targets, event_mask=np.ones(len(data), bool))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[1], data.date.iloc[2]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[3]]
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0


def test_no_trade_missing_execution_balance_preserves_anchor_and_later_upgrade():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6),
        "wealth": [1., .99, 1.01, .98, .98, .98], "open": [10., 10., 9.9, 10.1, 9.8, 9.8],
        "dividend": np.zeros(6), "previous_close": [np.nan, 10., 9.9, 10.1, 9.8, 9.8]})
    ledger = pd.DataFrame({"date": data.date.iloc[1:].to_numpy(), "shares_before": [0, 100, 100, 100, 0],
        "filled_quantity": [100, 0, 0, -100, 0], "shares_after": [100, np.nan, np.nan, 0, np.nan],
        "shares": [100, 100, 100, 0, 0]})
    f = reference_entry_frame(data, ledger, 1)
    result = staged_targets(f, [1, 1, 1, 1, 0, np.nan], 1)
    assert f.reference_holding.iloc[1:4].all()
    np.testing.assert_allclose(f.reference_entry_wealth_open.iloc[1:4], [1., 1., 1.])
    np.testing.assert_allclose(result.target.iloc[:5], [.5, .5, 1., 1., 0.])
    assert result.price_upgrade.iloc[2] and not result.price_upgrade.iloc[1]
    assert not f.reference_holding.iloc[4] and np.isnan(f.reference_entry_wealth_open.iloc[4])
