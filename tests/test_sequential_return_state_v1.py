"""核对双向触发、前日波动、完整时钟以及真实资金进出场。"""
import numpy as np
import pandas as pd
import pytest
from research.sequential_return_state_inputs_v1 import sequential_states, sequential_return_factors, sequential_return_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'volatility_window': 20, 'risk_target': .1,
    'drift_allowance': .5, 'alarm_threshold': 5.}


def fixture():
    returns = np.r_[np.tile([.001, -.001], 15), np.tile([.006, .002], 25),
        np.tile([-.006, -.002], 25), np.tile([.006, .002], 25)]
    data = prices(10*np.cumprod(1+returns))
    data['total_simple'] = (data.close+data.dividend)/data.previous_close-1
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[25]


def test_exact_threshold_reset_direction_retention_and_opposite_alarm():
    f = sequential_states([1., 1., 4.5, 0., -5.5, 5.5, 5.5])
    np.testing.assert_allclose(f.up_evidence_before_reset, [.5, 1., 5., 0., 0., 5., 5.])
    assert f.direction_trigger.tolist() == [0, 0, 1, 0, -1, 1, 1]
    assert f.positive_direction.tolist() == [0, 0, 1, 1, 0, 1, 1]
    assert f.loc[f.direction_trigger.ne(0), ['up_evidence_after_reset', 'down_evidence_after_reset']].eq(0).all().all()
    with pytest.raises(ValueError):
        sequential_states([np.inf])


def test_missing_resets_evidence_and_only_complete_previous_window_recovers():
    f = sequential_states([5.5, np.nan, 0., 4., 2.])
    assert f.iloc[1].isna().all() and f.positive_direction.tolist()[2:] == [0., 0., 1.]
    data, parents, start = fixture()
    data.loc[70, 'total_simple'] = np.nan
    factors = sequential_return_factors(data)
    assert factors.positive_direction.iloc[70:91].isna().all() and pd.notna(factors.positive_direction.iloc[91])
    frames, _ = sequential_return_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[70:91].isna().all()
    flat = data.copy(); flat['total_simple'] = 0.
    assert sequential_return_factors(flat).positive_direction.isna().all()


def test_previous_scale_excludes_today_and_cash_dividend_is_not_a_loss():
    data, _, _ = fixture()
    factors = sequential_return_factors(data)
    assert factors.previous_twenty_day_scale.iloc[30] == pytest.approx(data.total_simple.iloc[10:30].std(ddof=1))
    changed = data.copy(); changed.loc[30, 'total_simple'] = .09
    got = sequential_return_factors(changed)
    assert got.previous_twenty_day_scale.iloc[30] == factors.previous_twenty_day_scale.iloc[30]
    assert got.standardized_daily_return.iloc[30] == pytest.approx(.09/factors.previous_twenty_day_scale.iloc[30])
    raw = np.full(55, 10.); raw[30:] = 9.
    dividends = np.zeros(55); dividends[30] = 1.
    ex = prices(raw, dividends)
    ex['total_simple'] = (ex.close+ex.dividend)/ex.previous_close-1
    ex['vol20'] = 0.
    assert ex.total_simple.iloc[30] == 0 and sequential_return_factors(ex).direction_trigger.dropna().empty


def test_risk_sizing_known_zero_unknown_state_and_initial_trade_band():
    data, parents, start = fixture()
    data.loc[40:43, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = sequential_return_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[39] == .5 and target.iloc[40:42].isna().all()
    assert target.iloc[42] == .25 and target.iloc[43] == 1. and target.iloc[110] == 0.
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_prefix_future_isolation_and_identical_cost_independent_targets():
    data, parents, start = fixture()
    original, _ = sequential_return_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[120:, 'total_simple'] *= -1.
    future, _ = sequential_return_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:120], future['BASE'].iloc[:120])
    short, _ = sequential_return_frames(data.iloc[:120], parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:119], short['BASE'].iloc[:119])
    pd.testing.assert_frame_equal(original['BASE'].drop(columns='source_cost'), original['STRESS'].drop(columns='source_cost'))
    assert original['BASE'][PRIMARY+'_target'].iloc[:24].isna().all() and pd.isna(original['BASE'][PRIMARY+'_target'].iloc[-1])
    expected = pd.DatetimeIndex(data.date.iloc[24:-1])+pd.Timedelta(hours=15, minutes=5)
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[24:-1]).equals(expected)


def test_actual_next_open_exit_reentry_unknown_holding_and_dividend_rights():
    original, parents, start = fixture()
    raw = original.close.to_numpy().copy(); raw[52:] -= .1
    dividends = np.zeros(len(raw)); dividends[52] = .1
    data = prices(raw, dividends)
    data.loc[52, 'open'] = raw[51]-.1
    data['total_simple'] = (data.close+data.dividend)/data.previous_close-1
    data['vol20'] = .2
    data.loc[45, 'vol20'] = np.nan
    frames, _ = sequential_return_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    rights = pd.DataFrame({'record_date': [data.date.iloc[51]], 'ex_date': [data.date.iloc[52]],
        'payment_date': [data.date.iloc[55]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert ledger.filled_quantity.gt(0).sum() >= 2 and ledger.filled_quantity.lt(0).sum() >= 2
    assert decisions.loc[decisions.origin_index.eq(45), 'requested_quantity'].iloc[0] == 0
    earned = by_date.loc[data.date.iloc[51], 'shares']*.1
    assert earned > 0 and by_date.loc[data.date.iloc[52], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[55], 'dividend_paid'] == pytest.approx(earned)
    zero = np.flatnonzero((target == 0) & np.r_[False, target[:-1] > 0])
    assert len(zero) > 0 and all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zero)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
