"""核对六种价格顺序、熵、完整窗口与真实进出场。"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import entropy as scipy_entropy
from research.ordinal_entropy_inputs_v1 import three_day_patterns, normalized_entropy, ordinal_summary, confirmed_entropy_directions, ordinal_entropy_factors, ordinal_entropy_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'entropy_window': 60, 'pattern_order': 3,
    'momentum_window': 60, 'volatility_window': 20, 'entry_threshold': .9, 'exit_threshold': .95,
    'confirmation_closes': 2, 'risk_target': .1}


def fixture():
    data = prices(np.r_[np.linspace(8., 16., 180), np.linspace(15.98, 9., 120), np.linspace(9.02, 17., 120)])
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[125]


def test_original_paper_example_patterns_ties_and_entropy():
    counts, entropy = ordinal_summary([4., 7., 9., 10., 6., 11., 3.])
    np.testing.assert_array_equal(counts, [2, 0, 1, 0, 2, 0])
    assert entropy == pytest.approx(-(2*.4*np.log(.4)+.2*np.log(.2))/np.log(6.))
    assert entropy == pytest.approx(scipy_entropy(counts)/np.log(6.))
    assert normalized_entropy([1, 1, 1, 1, 1, 1]) == pytest.approx(1.)
    for values, code in [([1., 1., 2.], 0), ([2., 1., 1.], 3), ([1., 2., 1.], 1), ([1., 1., 1.], 0)]:
        assert three_day_patterns(values)[-1] == code
    for values in [np.arange(60.), -np.arange(60.), np.ones(60)]:
        assert ordinal_summary(values)[1] == 0.
    with pytest.raises(ValueError):
        ordinal_summary([1., np.nan, 2.])


def test_strict_threshold_separate_exit_counts_and_unknown_restart():
    entropies = [.9, .8, .8, .95, .96, .9, .96, .96, .8, np.nan, .8, .8, .8, .8]
    momentum = [.1, .1, .1, .1, .1, 0., .1, .1, .1, .1, .1, .1, 0., 0.]
    f = confirmed_entropy_directions(entropies, momentum)
    np.testing.assert_allclose(f.positive_direction, [0, 0, 1, 1, 1, 1, 1, 0, 0, np.nan, 0, 1, 1, 0], equal_nan=True)
    assert f.entropy_exit_confirmation_count.iloc[3] == 0 and f.entropy_exit_confirmation_count.iloc[7] == 2
    assert f.momentum_exit_confirmation_count.iloc[5] == 1 and f.momentum_exit_confirmation_count.iloc[6] == 0
    assert f.entry_confirmation_count.iloc[10] == 1


def test_sixty_sixtyone_missing_boundaries_flat_wealth_and_ex_dividend():
    data, parents, start = fixture()
    f = ordinal_entropy_factors(data)
    assert f.entropy60.iloc[:59].isna().all() and f.entropy60.iloc[59] == 0.
    assert f.momentum60.iloc[:60].isna().all() and f.momentum60.iloc[60] > 0.
    data.loc[150, 'wealth'] = np.nan
    changed = ordinal_entropy_factors(data)
    assert changed.entropy60.iloc[150:210].isna().all() and pd.notna(changed.entropy60.iloc[210])
    assert changed.momentum60.iloc[150:211].isna().all() and pd.notna(changed.momentum60.iloc[211])
    assert changed.positive_direction.iloc[210:211].isna().all()
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    ex = prices(raw, cash); ex['vol20'] = np.nan
    ex_factors = ordinal_entropy_factors(ex)
    assert ex_factors.entropy60.iloc[59:].eq(0.).all() and ex_factors.momentum60.iloc[60:].eq(0.).all()
    frames, _ = ordinal_entropy_frames(ex, parents, CFG, ex.date.iloc[125])
    assert frames['BASE'][PRIMARY+'_target'].iloc[124:-1].eq(0.).all()


def test_risk_cap_known_flat_and_initial_band_exception():
    data, parents, start = fixture()
    data.loc[140:143, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = ordinal_entropy_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[139] == .5 and target.iloc[140:142].isna().all()
    assert target.iloc[142] == .25 and target.iloc[143] == 1.
    data.loc[250, 'vol20'] = np.nan
    frames, _ = ordinal_entropy_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[250] == 0.
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_future_prefix_and_both_costs_preserve_target_clock():
    data, parents, start = fixture()
    original, _ = ordinal_entropy_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[300:, 'wealth'] *= 100.
    future, _ = ordinal_entropy_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:300], future['BASE'].iloc[:300])
    short, _ = ordinal_entropy_frames(data.iloc[:300], parents, CFG, start)
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
    frames, _ = ordinal_entropy_frames(data, parents, CFG, start)
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
