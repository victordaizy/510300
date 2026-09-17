"""核对上涨下跌计数、持平及完整窗口、未知与真实资金进出场。"""
import numpy as np
import pandas as pd
import pytest
from research.return_sign_balance_inputs_v1 import sign_balance_summary, confirmed_sign_directions, return_sign_balance_factors, return_sign_balance_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'sign_window': 60, 'momentum_window': 60,
    'volatility_window': 20, 'entry_threshold': 0., 'exit_threshold': 0., 'confirmation_closes': 2, 'risk_target': .1}


def fixture():
    data = prices(np.r_[np.linspace(8., 16., 180), np.linspace(15.98, 9., 120), np.linspace(9.02, 17., 120)])
    data['total_log'] = np.log((data.close+data.dividend)/data.previous_close)
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[125]


def test_hand_actual_up_down_flat_counts_and_same_frequency_different_momentum():
    a = np.r_[np.full(30, .01), np.full(20, -.005), np.zeros(10)]
    b = np.r_[np.full(30, .001), np.full(20, -.02), np.zeros(10)]
    for values in [a, b]:
        row = sign_balance_summary(values)
        assert row['up_days60'] == 30 and row['down_days60'] == 20 and row['flat_days60'] == 10
        assert row['sign_balance60'] == pytest.approx(1/6)
    assert a.sum() > 0 and b.sum() < 0
    f = confirmed_sign_directions([1/6, 1/6, 1/6, 1/6], [a.sum(), a.sum(), b.sum(), b.sum()])
    np.testing.assert_array_equal(f.positive_direction, [0, 1, 1, 0])
    assert sign_balance_summary(np.ones(60))['sign_balance60'] == 1.
    assert sign_balance_summary(-np.ones(60))['sign_balance60'] == -1.
    assert sign_balance_summary(np.zeros(60)) == {'up_days60': 0, 'down_days60': 0, 'flat_days60': 60, 'sign_balance60': 0.}
    with pytest.raises(ValueError):
        sign_balance_summary(np.r_[np.nan, np.zeros(59)])


def test_strict_entry_inclusive_exit_separate_counts_and_unknown_restart():
    scores = [0., .2, .2, .3, 0., .2, 0., 0., .2, np.nan, .2, .2, .2, .2]
    momentum = [.1, .1, .1, .1, .1, 0., .1, .1, .1, .1, .1, .1, 0., 0.]
    f = confirmed_sign_directions(scores, momentum)
    np.testing.assert_allclose(f.positive_direction, [0, 0, 1, 1, 1, 1, 1, 0, 0, np.nan, 0, 1, 1, 0], equal_nan=True)
    assert f.sign_exit_confirmation_count.iloc[4] == 1 and f.sign_exit_confirmation_count.iloc[7] == 2
    assert f.momentum_exit_confirmation_count.iloc[5] == 1 and f.momentum_exit_confirmation_count.iloc[6] == 0
    assert f.entry_confirmation_count.iloc[10] == 1


def test_complete_window_missing_reset_and_ex_dividend_flat_known_zero():
    data, parents, start = fixture()
    f = return_sign_balance_factors(data)
    assert f.sign_balance60.iloc[:60].isna().all() and pd.notna(f.sign_balance60.iloc[60])
    data.loc[150, 'total_log'] = np.nan
    changed = return_sign_balance_factors(data)
    assert changed.sign_balance60.iloc[150:210].isna().all() and pd.notna(changed.sign_balance60.iloc[210])
    frames, _ = return_sign_balance_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[150:210].isna().all()
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    ex = prices(raw, cash); ex['vol20'] = .2
    ex['total_log'] = np.log((ex.close+ex.dividend)/ex.previous_close)
    ex_factors = return_sign_balance_factors(ex)
    assert ex_factors.total_daily_log.iloc[125] == 0. and ex_factors.momentum60.iloc[60:].eq(0.).all()
    assert ex_factors.sign_balance60.iloc[60:].eq(0.).all() and ex_factors.flat_days60.iloc[60:].eq(60.).all()


def test_risk_cap_known_flat_and_initial_band_exception():
    data, parents, start = fixture()
    data.loc[140:143, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = return_sign_balance_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[139] == .5 and target.iloc[140:142].isna().all()
    assert target.iloc[142] == .25 and target.iloc[143] == 1.
    data.loc[250, 'vol20'] = np.nan
    frames, _ = return_sign_balance_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[250] == 0.
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_future_prefix_and_both_costs_preserve_target_clock():
    data, parents, start = fixture()
    original, _ = return_sign_balance_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[300:, 'total_log'] *= -1.
    future, _ = return_sign_balance_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:300], future['BASE'].iloc[:300])
    short, _ = return_sign_balance_frames(data.iloc[:300], parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:299], short['BASE'].iloc[:299])
    pd.testing.assert_frame_equal(original['BASE'].drop(columns='source_cost'), original['STRESS'].drop(columns='source_cost'))
    expected = pd.DatetimeIndex(data.date.iloc[124:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[124:-1]).equals(expected)
    assert original['BASE'][PRIMARY+'_target'].iloc[:124].isna().all() and pd.isna(original['BASE'][PRIMARY+'_target'].iloc[-1])


def test_actual_next_open_unknown_hold_exit_reentry_and_dividend_cash():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy(); raw[152:] -= .1
    cash = np.zeros(len(raw)); cash[152] = .1
    data = prices(raw, cash); data.loc[152, 'open'] = raw[151]-.1
    data['total_log'] = np.log((data.close+data.dividend)/data.previous_close)
    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan
    frames, _ = return_sign_balance_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    rights = pd.DataFrame({'record_date': [data.date.iloc[151]], 'ex_date': [data.date.iloc[152]],
        'payment_date': [data.date.iloc[155]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 2
    assert decisions.loc[decisions.origin_index.eq(145), 'requested_quantity'].iloc[0] == 0
    earned = by_date.loc[data.date.iloc[151], 'shares']*.1
    assert earned > 0 and by_date.loc[data.date.iloc[152], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[155], 'dividend_paid'] == pytest.approx(earned)
    zero = np.flatnonzero((target == 0.) & np.r_[False, target[:-1] > 0.])
    assert len(zero) > 0 and all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zero)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
