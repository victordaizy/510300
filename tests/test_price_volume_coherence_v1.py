"""核对价量相关、独立退出确认、未来隔离和真实资金进出场。"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from research.price_volume_coherence_inputs_v1 import pearson_summary, confirmed_coherence_directions, price_volume_coherence_factors, price_volume_coherence_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'correlation_window': 20, 'momentum_window': 20,
    'volatility_window': 20, 'entry_threshold': .2, 'exit_threshold': -.2, 'confirmation_closes': 2, 'risk_target': .1}


def fixture():
    data = prices(np.r_[np.linspace(8., 16., 180), np.linspace(15.98, 9., 120), np.linspace(9.02, 17., 120)])
    data['volume'] = 100000.*data.wealth**2
    data['volume_unit'] = 'share'
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[125]


def test_hand_calculated_correlation_constant_missing_and_scale():
    x, y = [1., 2., 4.], [1., 3., 2.]
    got = pearson_summary(x, y)
    assert got['centered_cross_sum20'] == pytest.approx(1.)
    assert got['return_centered_square_sum20'] == pytest.approx(14/3)
    assert got['volume_centered_square_sum20'] == 2.
    expected = 1/np.sqrt(28/3)
    assert got['price_volume_correlation20'] == pytest.approx(expected)
    assert got['price_volume_correlation20'] == pytest.approx(pearsonr(x, y).statistic)
    assert pearson_summary(np.array(x)*100, np.array(y)+500)['price_volume_correlation20'] == pytest.approx(expected)
    assert pearson_summary(x, -np.array(y))['price_volume_correlation20'] == pytest.approx(-expected)
    assert np.isnan(pearson_summary([1., 1., 1.], y)['price_volume_correlation20'])
    with pytest.raises(ValueError):
        pearson_summary([1., np.nan], [1., 2.])


def test_thresholds_separate_exit_counters_and_unknown_restart():
    correlations = [.2, .3, .3, -.2, -.3, .1, -.3, -.3, .3, np.nan, .3, .3, .3, .3]
    momentum = [.1, .1, .1, .1, .1, 0., .1, .1, .1, .1, .1, .1, 0., 0.]
    f = confirmed_coherence_directions(correlations, momentum)
    np.testing.assert_allclose(f.positive_direction, [0, 0, 1, 1, 1, 1, 1, 0, 0, np.nan, 0, 1, 1, 0], equal_nan=True)
    assert f.correlation_exit_confirmation_count.iloc[3] == 0 and f.correlation_exit_confirmation_count.iloc[7] == 2
    assert f.momentum_exit_confirmation_count.iloc[5] == 1 and f.momentum_exit_confirmation_count.iloc[6] == 0
    assert f.entry_confirmation_count.iloc[10] == 1


def test_volume_previous_day_units_missing_windows_and_dividend():
    data, parents, start = fixture()
    f = price_volume_coherence_factors(data)
    np.testing.assert_allclose(f.price_volume_correlation20.iloc[20:], 1., atol=1e-12)
    assert f.price_volume_correlation20.iloc[:20].isna().all()
    data.loc[150, 'volume'] = np.nan
    changed = price_volume_coherence_factors(data)
    assert changed.price_volume_correlation20.iloc[150:171].isna().all()
    assert pd.notna(changed.price_volume_correlation20.iloc[171]) and changed.positive_direction.iloc[171] == 0
    for column, value in [('volume', 0.), ('volume_unit', 'lot')]:
        bad = data.copy(); bad.loc[149, column] = value
        with pytest.raises(ValueError):
            price_volume_coherence_factors(bad)
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    ex = prices(raw, cash); ex['volume'] = np.linspace(100000., 200000., len(raw)); ex['volume_unit'] = 'share'; ex['vol20'] = .2
    ex_factors = price_volume_coherence_factors(ex)
    assert ex_factors.total_daily_log.iloc[125] == 0. and ex_factors.momentum20.iloc[20:].eq(0.).all()
    assert ex_factors.price_volume_correlation20.isna().all()


def test_risk_cap_known_flat_and_initial_band_exception():
    data, parents, start = fixture()
    data.loc[140:143, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = price_volume_coherence_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[139] == .5 and target.iloc[140:142].isna().all()
    assert target.iloc[142] == .25 and target.iloc[143] == 1.
    data.loc[250, 'vol20'] = np.nan
    frames, _ = price_volume_coherence_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[250] == 0.
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_future_prefix_and_both_costs_preserve_target_clock():
    data, parents, start = fixture()
    original, _ = price_volume_coherence_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[300:, 'volume'] *= 100.
    future, _ = price_volume_coherence_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:300], future['BASE'].iloc[:300])
    short, _ = price_volume_coherence_frames(data.iloc[:300], parents, CFG, start)
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
    data['volume'] = 100000.*data.wealth**2; data['volume_unit'] = 'share'
    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan
    frames, _ = price_volume_coherence_frames(data, parents, CFG, start)
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
