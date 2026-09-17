"""核对带符号排序、连续确认、波动仓位和真实进出场。"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, wilcoxon
from research.session_signed_rank_inputs_v1 import signed_rank_summary, confirmed_rank_directions, session_signed_rank_factors, session_signed_rank_frames, PRIMARY, CANDIDATES
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'volatility_window': 20, 'rank_window': 60, 'entry_threshold': 1.96, 'exit_threshold': 0., 'confirmation_closes': 2,
    'risk_target': .1}


def fixture():
    data = prices(np.r_[np.linspace(8., 16., 180), np.linspace(15.98, 9., 120), np.linspace(9.02, 17., 120)])
    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))
    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)
    data['vol20'] = .2
    return data, {cost: {} for cost in CFG['costs']}, data.date.iloc[125]


def test_known_signed_ranks_ties_zero_independent_score_and_extreme_rank():
    values = np.array([1., -2., 2., 0., 3.])
    got = signed_rank_summary(values)
    assert got['nonzero_count'] == 4 and got['positive_rank_sum'] == 7.5 and got['negative_rank_sum'] == 2.5
    assert got['rank_square_sum'] == 29.5 and got['rank_score60'] == pytest.approx(5/np.sqrt(29.5))
    ranks = rankdata(abs(values[values != 0]), method='average')
    expected = np.dot(np.sign(values[values != 0]), ranks)/np.linalg.norm(ranks)
    assert got['rank_score60'] == pytest.approx(expected)
    assert got['rank_score60'] == pytest.approx(wilcoxon(values, alternative='greater', method='asymptotic', zero_method='wilcox', correction=False).zstatistic)
    values[-1] = 3000.
    assert signed_rank_summary(values) == got
    assert signed_rank_summary(-values)['rank_score60'] == -got['rank_score60']
    assert signed_rank_summary(np.zeros(60))['rank_score60'] == 0.
    with pytest.raises(ValueError):
        signed_rank_summary([1., np.nan])


def test_strict_threshold_two_closes_state_retention_and_missing_reset():
    f = confirmed_rank_directions([1.96, 2., 2., 0., -.1, -.1, 2., np.nan, 2., 2.])
    np.testing.assert_allclose(f.positive_direction, [0, 0, 1, 1, 1, 0, 0, np.nan, 0, 1], equal_nan=True)
    np.testing.assert_allclose(f.positive_confirmation_count, [0, 1, 2, 0, 0, 0, 1, np.nan, 1, 2], equal_nan=True)
    data, parents, start = fixture()
    data.loc[150, 'intraday_log'] = np.nan
    f = session_signed_rank_factors(data)
    assert f.rank_score60.iloc[150:210].isna().all() and pd.notna(f.rank_score60.iloc[210])
    frames, _ = session_signed_rank_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[150:210].isna().all()


def test_ordinary_risk_cap_unknown_positive_and_known_flat_state():
    data, parents, start = fixture()
    data.loc[140:143, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = session_signed_rank_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[139] == .5 and target.iloc[140:142].isna().all()
    assert target.iloc[142] == .25 and target.iloc[143] == 1.
    flat = data.copy(); flat[['intraday_log', 'overnight_log']] = 0.; flat['vol20'] = np.nan
    frames, _ = session_signed_rank_frames(flat, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[124:-1].eq(0.).all()


def test_ex_dividend_session_difference_zero_and_initial_band_exception():
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    data = prices(raw, cash); data.loc[125, 'open'] = 9.
    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))
    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)
    data['vol20'] = .2
    f = session_signed_rank_factors(data)
    assert f.session_difference.iloc[125] == 0. and f.rank_score60.iloc[60:].eq(0.).all()
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


def test_future_prefix_and_both_costs_preserve_exact_known_target_clock():
    data, parents, start = fixture()
    original, _ = session_signed_rank_frames(data, parents, CFG, start)
    changed = data.copy(); changed.loc[300:, 'intraday_log'] *= -1.
    future, _ = session_signed_rank_frames(changed, parents, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:300], future['BASE'].iloc[:300])
    short, _ = session_signed_rank_frames(data.iloc[:300], parents, CFG, start)
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
    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))
    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)
    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan
    frames, _ = session_signed_rank_frames(data, parents, CFG, start)
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
