"""核对局部先后高点、持续回撤、双上限和真实进出场。"""
import numpy as np
import pandas as pd
import pytest
from research.drawdown_depth_risk_inputs_v1 import drawdown_depth_factors, drawdown_depth_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'volatility_window': 20, 'drawdown_window': 60,
    'risk_target': .1, 'drawdown_budget': .04}


def fixture():
    data = prices(np.r_[np.linspace(8., 16., 180), np.linspace(15.98, 9., 120), np.linspace(9.02, 17., 120)])
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[125]


def test_known_drawdown_depth_duration_order_recovery_and_scale_invariance():
    data = prices(np.r_[np.full(20, 10.), np.full(20, 9.), np.full(80, 10.)])
    data['vol20'] = .2
    f = drawdown_depth_factors(data)
    assert f.drawdown_strength60.iloc[:59].isna().all()
    assert f.drawdown_strength60.iloc[59] == pytest.approx(np.sqrt(20*.1**2/60))
    assert f.drawdown_strength60.iloc[99] == 0.
    changed = data.copy(); changed['wealth'] *= 3.7
    np.testing.assert_allclose(drawdown_depth_factors(changed).drawdown_strength60, f.drawdown_strength60, atol=1e-15, equal_nan=True)
    late_high = prices(np.r_[np.full(59, 9.), 10.]); late_high['vol20'] = .2
    assert drawdown_depth_factors(late_high).drawdown_strength60.iloc[59] == 0.


def test_missing_wealth_requires_complete_windows_and_invalid_inputs_stop():
    data, parents, start = fixture()
    data.loc[150, 'wealth'] = np.nan
    f = drawdown_depth_factors(data)
    assert f.drawdown_strength60.iloc[150:210].isna().all() and pd.notna(f.drawdown_strength60.iloc[210])
    assert f.trend_deviation120.iloc[150:270].isna().all() and pd.notna(f.trend_deviation120.iloc[270])
    frames, _ = drawdown_depth_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[150:270].isna().all()
    for column, value in [('wealth', 0.), ('wealth', np.inf), ('vol20', -.1)]:
        invalid = data.copy(); invalid.loc[10, column] = value
        with pytest.raises(ValueError):
            drawdown_depth_factors(invalid)


def test_two_caps_known_zero_and_unknown_positive_risk():
    data = prices(np.r_[np.full(60, 10.), np.full(20, 20.), np.full(40, 17.), np.full(10, 17.)])
    data['vol20'] = .05
    parents = {cost: {} for cost in CFG['costs']}
    frames, _ = drawdown_depth_frames(data, parents, CFG, data.date.iloc[120])
    f = frames['BASE']; expected = .04/np.sqrt(40*.15**2/60)
    assert f.trend_deviation120.iloc[119] > 0. and f[PRIMARY+'_target'].iloc[119] == pytest.approx(expected)
    data.loc[119:122, 'vol20'] = [.5, np.nan, 0., .2]
    frames, _ = drawdown_depth_frames(data, parents, CFG, data.date.iloc[120])
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[119] == .2 and target.iloc[120:122].isna().all()
    flat = prices(np.full(140, 10.)); flat['vol20'] = np.nan
    frames, _ = drawdown_depth_frames(flat, parents, CFG, flat.date.iloc[120])
    assert frames['BASE'][PRIMARY+'_target'].iloc[119:-1].eq(0.).all()


def test_ex_dividend_preserves_wealth_and_empty_account_bypasses_band():
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    data = prices(raw, cash); data['vol20'] = .2
    f = drawdown_depth_factors(data)
    assert f.drawdown_strength60.iloc[59:].eq(0.).all() and f.trend_deviation120.iloc[119:].eq(0.).all()
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_future_prefix_and_both_costs_preserve_exact_known_target_clock():
    data, parents, start = fixture()
    original, _ = drawdown_depth_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[300:, 'wealth'] *= 2.
    future, _ = drawdown_depth_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:300], future['BASE'].iloc[:300])
    short, _ = drawdown_depth_frames(data.iloc[:300], parents, CFG, start)
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
    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan
    frames, _ = drawdown_depth_frames(data, parents, CFG, start)
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
