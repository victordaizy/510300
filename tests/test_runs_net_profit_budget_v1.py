"""核对正均值预算、双零现金、月度时点和实际进退。"""
import numpy as np
import pandas as pd
import pytest
from research.runs_net_profit_budget_inputs_v1 import PRIMARY, MODELS, CANDIDATES, net_profit_budget, runs_net_profit_budget_frames
from research.event_clock_account_v1 import simulate_event_account
from tests.test_heikin_price_state_v1 import prices
from tests.test_runs_covariance_budget_v1 import fixture, ledgers
from tests.test_runs_covariance_budget_v1 import test_own_account_entry_band_positive_rounding_and_zero_exit as account_boundaries
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'profit_window': 242, 'initial_budgets': [.5, .5], 'budget_update': 'FIRST_TRADING_DAY_MONTH_CLOSE', 'budget_source_cost': 'BASE'}


def test_mean_positive_ratio_single_source_and_nonpositive_cash():
    for ma, mb, expected in [(.001, .003, (.25, .75)), (.001, -.002, (1., 0.)),
        (0., .002, (0., 1.)), (-.001, -.002, (0., 0.)), (0., 0., (0., 0.))]:
        row = net_profit_budget(np.full(242, ma), np.full(242, mb), (.5, .5))
        assert row['mean_reference'] == pytest.approx(ma) and row['mean_runs'] == pytest.approx(mb)
        assert (row['budget_reference'], row['budget_runs']) == pytest.approx(expected)
        assert row['status'] == ('CASH_NONPOSITIVE_MEANS' if sum(expected) == 0 else 'UPDATED_POSITIVE_MEAN')


def test_missing_and_incomplete_retain_both_known_budgets():
    for previous in [(.3, .7), (0., 0.)]:
        for a, b, expected in [(np.zeros(241), np.zeros(241), 'INSUFFICIENT_WINDOW'),
            (np.zeros(242), np.r_[np.nan, np.ones(241)], 'MISSING_RETURN')]:
            row = net_profit_budget(a, b, previous)
            assert row['status'] == expected and (row['budget_reference'], row['budget_runs']) == previous


def test_month_first_only_complete_window_two_costs_and_period_reset():
    data, parents, base, start = fixture()
    base = ledgers(data, 3, [np.full(len(data)-3, .001), np.full(len(data)-3, .003)])
    frames, summary, events = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    updates = [r for r in events if r['status'] == 'UPDATED_POSITIVE_MEAN']
    assert updates and all(r['sample_count'] == 242 and r['window_end'] == r['origin'] for r in updates)
    first_update = updates[0]['origin_index']
    assert frames['BASE'].budget_reference.iloc[2:first_update].eq(.5).all()
    np.testing.assert_allclose(frames['BASE'].budget_reference.iloc[first_update:-1], .25, atol=1e-15)
    for t in range(3, len(data)-1):
        if data.date.iloc[t].month == data.date.iloc[t-1].month:
            assert frames['BASE'].budget_reference.iloc[t] == frames['BASE'].budget_reference.iloc[t-1]
    pd.testing.assert_series_equal(frames['BASE'].budget_reference, frames['STRESS'].budget_reference)
    data, parents, base, start = fixture(first=220)
    reset, _, _ = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    assert reset['BASE'].budget_reference.iloc[219:-1].eq(.5).all()
    assert reset['BASE'][PRIMARY+'_target'].iloc[:219].isna().all() and pd.isna(reset['BASE'][PRIMARY+'_target'].iloc[-1])


def test_future_terminal_and_shorter_prefix_isolation():
    data, parents, base, start = fixture()
    original, _, _ = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    changed = {model: f.copy() for model, f in base.items()}
    for f in changed.values():
        f.loc[f.date.ge(data.date.iloc[350]), 'net_return'] = .03
        f['equity'] = CFG['initial_capital']*(1.+f.net_return).cumprod()
    future, _, _ = runs_net_profit_budget_frames(data, parents, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:350], future['BASE'].iloc[:350])
    terminal = {model: f.copy() for model, f in base.items()}
    for f in terminal.values():
        f.loc[len(f)-1, 'net_return'] = .9
        f.loc[len(f)-1, 'equity'] = f.equity.iloc[-2]*1.9
    last, _, _ = runs_net_profit_budget_frames(data, parents, terminal, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'], last['BASE'])
    prefix_parents = {cost: {model: f[f.origin_index.lt(329)].copy() for model, f in group.items()} for cost, group in parents.items()}
    prefix_base = {model: f[f.date.le(data.date.iloc[329])].copy() for model, f in base.items()}
    for f in prefix_base.values():
        f.loc[f.index[-1], 'mark_clock'] = 'OPEN_TERMINAL'
    short, _, _ = runs_net_profit_budget_frames(data.iloc[:330], prefix_parents, prefix_base, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:329], short['BASE'].iloc[:329])


def test_zero_budget_unknown_parent_and_wrong_cost_rejected():
    data, parents, base, start = fixture()
    base = ledgers(data, 3, [np.full(len(data)-3, -.001), np.full(len(data)-3, .002)])
    frames, _, events = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    t = next(r['origin_index'] for r in events if r['status'] == 'UPDATED_POSITIVE_MEAN')
    assert frames['BASE'].budget_reference.iloc[t] == 0.
    assert frames['BASE'][PRIMARY+'_target'].iloc[t] == .2 and frames['STRESS'][PRIMARY+'_target'].iloc[t] == .1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(t), 'reference_weight'] = np.nan
    unknown, _, _ = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    assert pd.isna(unknown['BASE'][PRIMARY+'_target'].iloc[t])
    base[MODELS[0]]['source_cost'] = 'STRESS'
    with pytest.raises(ValueError):
        runs_net_profit_budget_frames(data, parents, base, CFG, start)


def test_actual_profit_budget_cash_exit_reentry_unknown_dividend_and_terminal():
    account_boundaries()
    original, parents, base, start = fixture(n=600)
    raw = original.close.to_numpy().copy(); raw[9:] -= .1
    div = np.zeros(len(raw)); div[9] = .1
    data = prices(raw, div); data.loc[9, 'open'] = raw[8]-.1
    returns = np.full(len(data)-3, .001)
    returns[250:310] = -.02
    base = ledgers(data, 3, [returns, returns])
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(8), 'reference_weight'] = np.nan
    frames, _, events = runs_net_profit_budget_frames(data, parents, base, CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    assert any(r['status'] == 'CASH_NONPOSITIVE_MEANS' for r in events)
    rights = pd.DataFrame({'record_date': [data.date.iloc[8]], 'ex_date': [data.date.iloc[9]],
        'payment_date': [data.date.iloc[11]], 'cash_dividend_per_share': [.1]})
    ledger, decisions = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], start, PRIMARY,
        targets=target, event_mask=np.ones(len(data), bool))
    by_date = ledger.set_index('date')
    assert by_date.loc[data.date.iloc[9], 'shares'] == by_date.loc[data.date.iloc[8], 'shares'] > 0
    earned = by_date.loc[data.date.iloc[8], 'shares']*.1
    assert by_date.loc[data.date.iloc[9], 'dividend_recognized'] == pytest.approx(earned)
    assert by_date.loc[data.date.iloc[11], 'dividend_paid'] == pytest.approx(earned)
    zeros = np.flatnonzero((target == 0.) & np.r_[False, target[:-1] > 0.])
    assert len(zeros) > 0 and all(by_date.loc[data.date.iloc[t+1], 'shares'] == 0 for t in zeros)
    assert by_date.loc[data.date.iloc[590], 'shares'] > 0
    assert ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == data.open.iloc[-1]
    assert ledger.accounting_error.abs().max() < 1e-6 and ledger.commission.sum() > 0 and ledger.slippage_cost.sum() > 0
    assert pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date))
