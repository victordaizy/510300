"""检查两次确认、取消等待、未知打断及原账户行为。"""
import numpy as np
import pandas as pd
import pytest

from research.event_clock_account_v1 import simulate_event_account
from research.two_close_zero_exit_account_v1 import simulate_confirmed_zero
from research.two_close_zero_exit_inputs_v1 import PRIMARY, zero_streak
from tests.test_target_band_partial_rebalance_v1 import fixture as base_fixture


def fixture():
    frame, dividends, cfg, cost = base_fixture()
    cfg['zero_confirmations'] = 2
    return frame, dividends, cfg, cost


def run(targets, frame=None, dividends=None):
    base, div, cfg, cost = fixture()
    frame, dividends = (base if frame is None else frame), (div if dividends is None else dividends)
    return simulate_confirmed_zero(frame, dividends, cfg, cost, str(frame.date.iloc[1].date()), PRIMARY,
        targets=np.asarray(targets, float), event_mask=np.ones(len(frame), bool))


def test_zero_counts_reset_on_positive_or_unknown_and_are_past_only():
    values = np.array([np.nan, 0., 0., .3, 0., np.nan, 0., 0., 0.])
    np.testing.assert_array_equal(zero_streak(values), [0, 1, 2, 0, 1, 0, 1, 2, 3])
    changed = values.copy()
    changed[5:] = .8
    np.testing.assert_array_equal(zero_streak(values)[:5], zero_streak(changed)[:5])
    with pytest.raises(ValueError, match='无效'):
        zero_streak([.3, 1.1])


def test_first_zero_keeps_shares_second_zero_exits_and_positive_reenters():
    ledger, decisions = run([.3, 0., 0., .4, .8, .8, 0., 0., 0., 0.])
    assert ledger.iloc[0].filled_quantity == 3000
    assert ledger.iloc[1].shares == 3000 and ledger.iloc[1].requested_quantity == 0
    assert ledger.iloc[2].filled_quantity == -3000 and ledger.iloc[2].shares == 0
    assert decisions.iloc[1].reference_weight == 0 and decisions.iloc[1].waiting_zero_confirmation == 1.
    assert ledger.iloc[3].filled_quantity > 0 and ledger.iloc[4].filled_quantity > 0
    assert ledger.iloc[-1].shares == 0


def test_recovered_positive_cancels_waiting_and_unknown_interrupts_confirmations():
    ledger, decisions = run([.3, 0., .3, .3, 0., np.nan, 0., 0., 0., 0.])
    assert ledger.iloc[:7].shares.eq(3000).all()
    assert decisions.iloc[2].used_zero_count == 0 and decisions.iloc[6].used_zero_count == 1
    assert decisions.iloc[6].waiting_zero_confirmation == 1.
    assert ledger.iloc[7].filled_quantity == -3000


def test_blocked_confirmed_exit_uses_next_recovered_positive_target():
    frame = fixture()[0]
    frame.loc[3, 'open'] = 9.
    ledger, decisions = run([.3, 0., 0., .3, 0., 0., 0., 0., 0., 0.], frame)
    assert ledger.iloc[2].status == 'UNFILLED_DIRECTIONAL_LIMIT'
    assert ledger.iloc[3].shares == 3000 and ledger.iloc[3].requested_quantity == 0
    assert decisions.iloc[3].used_zero_count == 0
    assert ledger.iloc[5].filled_quantity == -3000


def test_dividend_and_terminal_override_first_zero_waiting():
    frame = fixture()[0]
    frame.loc[3:, ['open', 'close']] = 9.9
    frame.loc[4:, 'previous_close'] = 9.9
    frame.loc[3, 'dividend'] = .1
    dividends = pd.DataFrame({'record_date': [frame.date.iloc[2]], 'ex_date': [frame.date.iloc[3]],
        'payment_date': [frame.date.iloc[5]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = run([.4]*8+[0., 0.], frame, dividends)
    assert ledger.iloc[2].dividend_recognized == pytest.approx(400.)
    assert ledger.iloc[4].dividend_paid == pytest.approx(400.)
    assert decisions.iloc[-1].waiting_zero_confirmation == 1.
    assert ledger.iloc[-1].shares == 0 and ledger.iloc[-1].requested_quantity == -4000
    assert ledger.accounting_error.abs().max() < 1e-6


def test_future_targets_and_prices_do_not_change_earlier_requests():
    targets = np.array([.3, 0., .3, .8, .2, .4, .8, .6, .3, .3])
    old_ledger, old_decisions = run(targets)
    frame = fixture()[0]
    frame.loc[6:, ['open', 'close', 'previous_close']] = 10.2
    targets[5:] = [0., np.nan, 0., 0., .5]
    new_ledger, new_decisions = run(targets, frame)
    pd.testing.assert_frame_equal(old_ledger.iloc[:5], new_ledger.iloc[:5])
    pd.testing.assert_frame_equal(old_decisions.iloc[:5], new_decisions.iloc[:5])


def test_all_positive_requests_reproduce_original_account_exactly():
    frame, dividends, cfg, cost = fixture()
    values = np.array([.3, .8, .3, .2, 1., .5, .1, .3, .5, .5])
    args = (frame, dividends, cfg, cost, str(frame.date.iloc[1].date()), PRIMARY)
    kwargs = {'targets': values, 'event_mask': np.ones(len(frame), bool)}
    original_ledger, original_decisions = simulate_event_account(*args, **kwargs)
    actual_ledger, actual_decisions = simulate_confirmed_zero(*args, **kwargs)
    pd.testing.assert_frame_equal(original_ledger, actual_ledger)
    pd.testing.assert_frame_equal(original_decisions, actual_decisions[original_decisions.columns])
