"""覆盖部分调仓、完整进入退出、实际状态以及共用账户的记账一致性。"""
import numpy as np
import pandas as pd
import pytest

from research.adaptive_allocation_v1 import Account, target_request
from research.event_account_request_callback_v1 import simulate_request_account
from research.event_clock_account_v1 import simulate_event_account
from research.target_band_partial_rebalance_inputs_v1 import CANDIDATES, partial_request


def fixture():
    dates = pd.bdate_range('2020-01-01', periods=10)
    frame = pd.DataFrame({'date': dates, 'open': 10., 'close': 10., 'previous_close': 10., 'dividend': 0., 'variance60': .0004})
    dividends = pd.DataFrame(columns=['record_date', 'ex_date', 'payment_date', 'cash_dividend_per_share'])
    cfg = {'initial_capital': 100000., 'lot': 100, 'tick': .001, 'limit_fraction': .1, 'weight_band': .1}
    cost = {'commission': .0002, 'minimum': 5., 'slippage': .0005}
    return frame, dividends, cfg, cost


def run(model, targets, frame=None, dividends=None):
    base, div, cfg, cost = fixture()
    frame, dividends = (base if frame is None else frame), (div if dividends is None else dividends)
    return simulate_request_account(frame, dividends, cfg, cost, str(frame.date.iloc[1].date()), model,
        targets=np.asarray(targets, float), event_mask=np.ones(len(frame), bool), request_policy=partial_request)


@pytest.mark.parametrize('model', CANDIDATES)
def test_entry_band_zero_unknown_and_invalid_target(model):
    account, cfg = Account(100000.), fixture()[2]
    assert partial_request(account, 10., .05, cfg, model)['requested_quantity'] == 500
    account.shares, account.cash = 2000, 80000.
    assert partial_request(account, 10., .25, cfg, model)['requested_quantity'] == 0
    assert partial_request(account, 10., 0., cfg, model)['requested_quantity'] == -2000
    assert partial_request(account, 10., np.nan, cfg, model)['requested_quantity'] == 0
    with pytest.raises(ValueError, match='超出范围'):
        partial_request(account, 10., 1.1, cfg, model)


@pytest.mark.parametrize('model', CANDIDATES)
def test_both_adjustment_directions_and_exact_boundary(model):
    account, cfg = Account(100000.), fixture()[2]
    account.shares, account.cash = 2000, 80000.
    up = partial_request(account, 10., .8, cfg, model)
    assert up['requested_quantity'] == (5000 if model == 'TARGET_BAND_EDGE' else 3000)
    assert up['reference_weight'] == .8 and up['application_weight'] != .8
    edge = partial_request(account, 10., .1, cfg, model)
    assert edge['requested_quantity'] == (0 if model == 'TARGET_BAND_EDGE' else -500)
    account.shares, account.cash = 8000, 20000.
    down = partial_request(account, 10., .05, cfg, model)
    assert down['requested_quantity'] == (-6500 if model == 'TARGET_BAND_EDGE' else -3800)


@pytest.mark.parametrize('model', CANDIDATES)
def test_actual_partial_buy_unknown_hold_full_exit_and_terminal(model):
    ledger, decisions = run(model, [.3, .8, np.nan, 0., 0., .4, .8, .8, np.nan, np.nan])
    assert ledger.iloc[0].filled_quantity == 3000
    assert ledger.iloc[1].filled_quantity > 0
    assert ledger.iloc[2].filled_quantity == 0
    assert ledger.iloc[3].shares == 0 and ledger.iloc[3].filled_quantity == -ledger.iloc[2].shares
    assert pd.isna(decisions.iloc[2].reference_weight)
    assert pd.isna(decisions.iloc[-1].reference_weight) and ledger.iloc[-1].shares == 0
    assert ledger.iloc[-1].requested_quantity == -ledger.iloc[-2].shares
    assert ledger.accounting_error.abs().max() < 1e-6


@pytest.mark.parametrize('model', CANDIDATES)
def test_blocked_exit_does_not_override_next_recovered_target(model):
    frame = fixture()[0]
    frame.loc[2, 'open'] = 9.
    ledger, decisions = run(model, [.3, 0., .3, 0., 0., 0., 0., 0., 0., 0.], frame)
    assert ledger.iloc[1].status == 'UNFILLED_DIRECTIONAL_LIMIT'
    assert ledger.iloc[2].shares == 3000 and ledger.iloc[2].requested_quantity == 0
    assert decisions.iloc[2].reference_weight == .3
    assert ledger.iloc[3].shares == 0


@pytest.mark.parametrize('model', CANDIDATES)
def test_dividend_receivable_then_payment_and_unknown_terminal(model):
    frame = fixture()[0]
    frame.loc[3:, ['open', 'close']] = 9.9
    frame.loc[4:, 'previous_close'] = 9.9
    frame.loc[3, 'dividend'] = .1
    dividends = pd.DataFrame({'record_date': [frame.date.iloc[2]], 'ex_date': [frame.date.iloc[3]],
        'payment_date': [frame.date.iloc[5]], 'cash_dividend_per_share': [.1]})
    ledger, _ = run(model, [.4]*8+[np.nan, np.nan], frame, dividends)
    assert ledger.iloc[2].dividend_recognized == pytest.approx(400.)
    assert ledger.iloc[2].dividend_receivable == pytest.approx(400.)
    assert ledger.iloc[4].dividend_paid == pytest.approx(400.)
    assert ledger.iloc[-1].shares == 0 and ledger.cash.ge(0).all()


@pytest.mark.parametrize('model', CANDIDATES)
def test_future_prices_and_targets_do_not_change_earlier_fills(model):
    target = np.array([.3, .8, .4, .8, .2, .4, .8, .6, .3, .3])
    original_ledger, original_decisions = run(model, target)
    frame = fixture()[0]
    frame.loc[6:, ['open', 'close', 'previous_close']] = 10.2
    target[5:] = [0., .8, np.nan, .01, .9]
    changed_ledger, changed_decisions = run(model, target, frame)
    pd.testing.assert_frame_equal(original_ledger.iloc[:5], changed_ledger.iloc[:5])
    pd.testing.assert_frame_equal(original_decisions.iloc[:5], changed_decisions.iloc[:5])


@pytest.mark.parametrize('model', CANDIDATES)
def test_new_account_with_original_request_reproduces_original_account(model):
    frame, dividends, cfg, cost = fixture()
    frame.loc[1, 'open'] = 11.
    targets = np.array([.3, .8, .4, .8, .2, np.nan, .8, 0., .3, .3])
    args = (frame, dividends, cfg, cost, str(frame.date.iloc[1].date()), model)
    kwargs = {'targets': targets, 'event_mask': np.ones(len(frame), bool)}
    original = simulate_event_account(*args, **kwargs)
    copied = simulate_request_account(*args, **kwargs, request_policy=lambda account, price, target, settings, _model: target_request(account, price, target, settings))
    for left, right in zip(original, copied):
        pd.testing.assert_frame_equal(left, right)
