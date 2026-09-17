"""核对完整风险窗口、预算时点、费用来源及真实资金进出。"""
import numpy as np
import pandas as pd
import pytest
from research.runs_covariance_budget_inputs_v1 import (
    PRIMARY, MODELS, CANDIDATES, covariance_budget, runs_covariance_budget_frames)
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import Account, target_request
from tests.test_heikin_price_state_v1 import prices
from tests.test_runs_reference_blend_v1 import fixture as small_fixture
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'risk_window': 242, 'initial_budget_reference': .5,
    'budget_update': 'FIRST_TRADING_DAY_MONTH_CLOSE', 'risk_source_cost': 'BASE'}


def ledgers(data, first, returns=None):
    count = len(data)-first
    default = [np.resize([-.001, 0., .001, 0.], count), np.resize([0., -.002, 0., .002], count)]
    result = {}
    for j, model in enumerate(MODELS):
        daily = default[j] if returns is None else returns[j]
        result[model] = pd.DataFrame({'date': data.date.iloc[first:].to_numpy(),
            'equity': CFG['initial_capital']*np.cumprod(1.+daily), 'net_return': daily,
            'mark_clock': ['CLOSE']*(count-1)+['OPEN_TERMINAL'], 'source_cost': 'BASE', 'source_model': model})
    return result


def fixture(n=430, first=3):
    data = prices(np.full(n, 10.))
    indices = np.arange(first-1, n-1)
    parents = {cost: {model: pd.DataFrame({'origin': data.date.iloc[indices].to_numpy(),
        'execution_date': data.date.iloc[indices+1].to_numpy(), 'origin_index': indices,
        'reference_weight': ([.8, .2] if cost == 'BASE' else [.4, .1])[j],
        'source_cost': cost, 'source_model': model}) for j, model in enumerate(MODELS)} for cost in CFG['costs']}
    return data, parents, ledgers(data, first), data.date.iloc[first]


def test_hand_covariance_minimum_and_clipped_bounds():
    a = np.r_[np.tile([-.001, 0., .001, 0.], 60), 0., 0.]
    b = np.r_[np.tile([0., -.002, 0., .002], 60), 0., 0.]
    row = covariance_budget(a, b, .5)
    assert row['status'] == 'UPDATED' and row['budget_reference'] == pytest.approx(.8)
    assert row['variance_reference'] == pytest.approx(.000120/241)
    assert row['variance_runs'] == pytest.approx(.000480/241)
    assert row['covariance'] == pytest.approx(0., abs=1e-20)
    assert row['difference_variance'] == pytest.approx(.000600/241)
    best = np.var(.8*a+.2*b, ddof=1)
    assert all(np.var(w*a+(1-w)*b, ddof=1) > best for w in [.0, .5, .79, .81, 1.])
    assert covariance_budget(a, -2*a, .5)['budget_reference'] == pytest.approx(2/3)
    assert covariance_budget(a, 2*a, .5)['budget_reference'] == 1.
    assert covariance_budget(2*a, a, .5)['budget_reference'] == 0.


def test_incomplete_missing_identical_and_zero_keep_previous():
    a = np.tile([-.001, .001], 121)
    for left, right, status in [(a[:-1], a[:-1], 'INSUFFICIENT_WINDOW'),
        (a, a, 'DEGENERATE_RISK'), (a, np.zeros(242), 'DEGENERATE_RISK'),
        (a, np.full(242, .001), 'DEGENERATE_RISK'),
        (a, np.r_[np.nan, a[1:]], 'MISSING_RETURN')]:
        row = covariance_budget(left, right, .37)
        assert row['status'] == status and row['budget_reference'] == .37


def test_month_start_only_complete_window_shared_budget_and_period_reset():
    data, parents, base, start = fixture()
    frames, summary, events = runs_covariance_budget_frames(data, parents, base, CFG, start)
    f = frames['BASE']
    updated = [r for r in events if r['status'] == 'UPDATED']
    assert updated and all(r['sample_count'] == 242 for r in updated)
    assert all(r['window_start'] == data.date.iloc[r['origin_index']-241] and
        r['window_end'] == data.date.iloc[r['origin_index']] for r in updated)
    first_update = updated[0]['origin_index']
    assert first_update >= 244 and f.budget_reference.iloc[2:first_update].eq(.5).all()
    for t in range(3, len(data)-1):
        if data.date.iloc[t].month == data.date.iloc[t-1].month:
            assert f.budget_reference.iloc[t] == f.budget_reference.iloc[t-1]
            assert f.budget_status.iloc[t] == 'MONTHLY_BUDGET_HELD'
    pd.testing.assert_series_equal(f.budget_reference, frames['STRESS'].budget_reference)
    assert summary[0]['valid_risk_updates'] == len(updated)
    later_data, later_parents, later_base, later_start = fixture(first=220)
    reset, _, later_events = runs_covariance_budget_frames(later_data, later_parents, later_base, CFG, later_start)
    assert reset['BASE'].budget_reference.iloc[219:-1].eq(.5).all()
    assert not any(r['status'] == 'UPDATED' for r in later_events)
    assert f[PRIMARY+'_target'].iloc[:2].isna().all() and pd.isna(f[PRIMARY+'_target'].iloc[-1])
    assert pd.DatetimeIndex(f.decision_time.iloc[2:-1]).equals(pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5))


def test_future_terminal_and_prefix_cannot_change_past():
    data, parents, base, start = fixture()
    original, _, _ = runs_covariance_budget_frames(data, parents, base, CFG, start)
    modified = {model: ledger.copy() for model, ledger in base.items()}
    for ledger in modified.values():
        ledger.loc[ledger.date.ge(data.date.iloc[350]), 'net_return'] = .03
        ledger['equity'] = CFG['initial_capital']*(1.+ledger.net_return).cumprod()
    future, _, _ = runs_covariance_budget_frames(data, parents, modified, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:350], future['BASE'].iloc[:350])
    terminal = {model: ledger.copy() for model, ledger in base.items()}
    for ledger in terminal.values():
        ledger.loc[len(ledger)-1, 'net_return'] = .9
        ledger.loc[len(ledger)-1, 'equity'] = ledger.equity.iloc[-2]*1.9
    end, _, _ = runs_covariance_budget_frames(data, parents, terminal, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'], end['BASE'])
    prefix_parents = {cost: {model: f[f.origin_index.lt(329)].copy() for model, f in group.items()} for cost, group in parents.items()}
    prefix_base = {model: f[f.date.le(data.date.iloc[329])].copy() for model, f in base.items()}
    for f in prefix_base.values():
        f.loc[f.index[-1], 'mark_clock'] = 'OPEN_TERMINAL'
    short, _, _ = runs_covariance_budget_frames(data.iloc[:330], prefix_parents, prefix_base, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:329], short['BASE'].iloc[:329])


def test_cost_identity_unknown_zero_budget_and_missing_risk():
    data, parents, base, start = fixture()
    b = np.resize([-.001, .001], len(data)-3)
    base = ledgers(data, 3, [2*b, b])
    frames, _, events = runs_covariance_budget_frames(data, parents, base, CFG, start)
    t = next(r['origin_index'] for r in events if r['status'] == 'UPDATED')
    assert frames['BASE'].budget_reference.iloc[t] == 0.
    assert frames['BASE'][PRIMARY+'_target'].iloc[t] == .2 and frames['STRESS'][PRIMARY+'_target'].iloc[t] == .1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(t), 'reference_weight'] = np.nan
    unknown, _, _ = runs_covariance_budget_frames(data, parents, base, CFG, start)
    assert pd.isna(unknown['BASE'][PRIMARY+'_target'].iloc[t])
    base[MODELS[0]].loc[base[MODELS[0]].date.eq(data.date.iloc[t]), 'net_return'] = np.nan
    missing, _, rows = runs_covariance_budget_frames(data, parents, base, CFG, start)
    assert next(r for r in rows if r['origin_index'] == t)['status'] == 'MISSING_RETURN'
    assert missing['BASE'].budget_reference.iloc[t] == .5
    base[MODELS[0]]['source_cost'] = 'STRESS'
    with pytest.raises(ValueError):
        runs_covariance_budget_frames(data, parents, base, CFG, start)
    data, parents, base, start = fixture()
    parents['STRESS'][MODELS[1]]['execution_date'] = parents['STRESS'][MODELS[1]].origin
    with pytest.raises(ValueError):
        runs_covariance_budget_frames(data, parents, base, CFG, start)


def test_own_account_entry_band_positive_rounding_and_zero_exit():
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == -10000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000
    account.cash, account.shares = 199000., 100
    assert target_request(account, 10., .0001, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -100


def test_actual_next_open_exit_unknown_hold_reentry_dividend():
    original, parents, start = small_fixture()
    raw = original.close.to_numpy().copy()
    raw[9:] -= .1
    cash = np.zeros(len(raw)); cash[9] = .1
    data = prices(raw, cash); data.loc[9, 'open'] = raw[8]-.1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(8), 'reference_weight'] = np.nan
    frames, _, _ = runs_covariance_budget_frames(data, parents, ledgers(data, 3), CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert by_date.loc[data.date.iloc[5], 'shares'] > 0 and by_date.loc[data.date.iloc[6], 'shares'] == 0
    assert by_date.loc[data.date.iloc[8], 'shares'] > 0 and by_date.loc[data.date.iloc[9], 'shares'] == by_date.loc[data.date.iloc[8], 'shares']
    assert decisions.loc[decisions.origin_index.eq(8), 'requested_quantity'].iloc[0] == 0
    earned = by_date.loc[data.date.iloc[8], 'shares']*.1
    assert earned > 0 and by_date.loc[data.date.iloc[9], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[11], 'dividend_paid'] == pytest.approx(earned)
    zero = np.flatnonzero((target == 0.) & np.r_[False, target[:-1] > 0.])
    assert len(zero) > 0 and all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zero)
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
