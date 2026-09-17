"""核对成熟周期、已确认分红、固定最近周期和实际账户进退。"""
import numpy as np
import pandas as pd
import pytest
from research.runs_closed_cycle_budget_inputs_v1 import (
    PRIMARY, MODELS, CANDIDATES, mature_saved_cycles, efficiency_score, cycle_budget_at, runs_closed_cycle_budget_frames)
from research.event_clock_account_v1 import simulate_event_account
from tests.test_runs_covariance_budget_v1 import fixture
from tests.test_runs_covariance_budget_v1 import test_own_account_entry_band_positive_rounding_and_zero_exit as account_boundaries
from tests.test_heikin_price_state_v1 import prices
from tests.test_trend_reference_router_v1 import CFG as OLD_CFG

CFG = {**OLD_CFG, 'candidate_models': list(CANDIDATES), 'parent_models': MODELS,
    'cycle_window': 20, 'minimum_mature_cycles': 5, 'initial_budgets': [.5, .5],
    'budget_update': 'FIRST_TRADING_DAY_MONTH_CLOSE', 'budget_source_cost': 'BASE'}


def cycle_sources(data, first=3, count=30, spacing=7, values=None):
    indices = first+1+np.arange(count)*spacing
    assert indices[-1] < len(data)
    result = {}
    for j, model in enumerate(MODELS):
        returns = np.resize([.04, .02, -.01, .02, -.01], count) if j == 0 else np.full(count, .02)
        if values is not None:
            returns = np.asarray(values[j], float)
        result[model] = pd.DataFrame({'source_model': model, 'cost': 'BASE', 'cycle': np.arange(count)+1,
            'entry_date': data.date.iloc[indices-1].to_numpy(), 'exit_date': data.date.iloc[indices].to_numpy(),
            'maturity_date': data.date.iloc[indices].to_numpy(), 'complete_information': True, 'cycle_net_return': returns})
    return result


def delayed_dividend_cycle():
    raw = np.full(18, 10.); raw[8:] -= .1
    div = np.zeros(18); div[8] = .1
    data = prices(raw, div); data.loc[8, 'open'] = 9.9
    targets = np.full(18, np.nan); targets[2:-1] = 0.; targets[2:5] = [.3, .6, .3]
    rights = pd.DataFrame({'record_date': [data.date.iloc[5]], 'ex_date': [data.date.iloc[8]],
        'payment_date': [data.date.iloc[10]], 'cash_dividend_per_share': [.1]})
    ledger, _ = simulate_event_account(data, rights, CFG, CFG['costs']['BASE'], data.date.iloc[3], PRIMARY,
        targets=targets, event_mask=np.ones(len(data), bool))
    ledger = ledger.assign(source_model=MODELS[0], source_cost='BASE')
    buys = ledger[ledger.filled_quantity.gt(0)]
    debit = float((buys.notional+buys.commission).sum())
    profit = float(ledger.equity.iloc[-1]-CFG['initial_capital'])
    saved = pd.DataFrame([{'source_model': MODELS[0], 'cost': 'BASE', 'cycle': 1,
        'entry_date': data.date.iloc[3], 'exit_date': data.date.iloc[6], 'terminal_exit': False,
        'buy_debit': debit, 'gross_price_profit': 0., 'commission': ledger.commission.sum(),
        'slippage': ledger.slippage_cost.sum(), 'dividend_recognized': ledger.dividend_recognized.sum(),
        'net_profit': profit, 'cycle_net_return': profit/debit}])
    return data, ledger, saved, rights


def test_actual_partial_trades_cycle_denominator_and_delayed_ex_dividend_maturity():
    data, ledger, saved, rights = delayed_dividend_cycle()
    checked = mature_saved_cycles(data, ledger, saved, rights, MODELS[0], 3)
    assert ledger.filled_quantity.gt(0).sum() == 2 and ledger.filled_quantity.lt(0).sum() == 2
    assert checked.maturity_date.iloc[0] == data.date.iloc[8] > checked.exit_date.iloc[0]
    assert checked.maturity_date.iloc[0] < rights.payment_date.iloc[0]
    assert checked.complete_information.iloc[0] and checked.dividend_events.iloc[0] == 1
    assert checked.cycle_net_return.iloc[0] == pytest.approx(saved.net_profit.iloc[0]/saved.buy_debit.iloc[0])
    assert checked.cycle_net_return.iloc[0] != pytest.approx(saved.net_profit.iloc[0]/CFG['initial_capital'])


def test_missing_cycle_information_and_wrong_cash_cost_or_exit_are_not_hidden():
    data, ledger, saved, rights = delayed_dividend_cycle()
    missing = saved.copy(); missing.loc[0, 'net_profit'] = np.nan
    checked = mature_saved_cycles(data, ledger, missing, rights, MODELS[0], 3)
    assert not checked.complete_information.iloc[0] and pd.isna(checked.cycle_net_return.iloc[0])
    wrong = saved.copy(); wrong.loc[0, 'buy_debit'] += 5
    with pytest.raises(ValueError):
        mature_saved_cycles(data, ledger, wrong, rights, MODELS[0], 3)
    wrong = saved.copy(); wrong.loc[0, 'exit_date'] = data.date.iloc[5]
    with pytest.raises(ValueError):
        mature_saved_cycles(data, ledger, wrong, rights, MODELS[0], 3)
    with pytest.raises(ValueError):
        mature_saved_cycles(data, ledger.assign(source_cost='STRESS'), saved, rights, MODELS[0], 3)


def test_hand_efficiency_recent_twenty_minimum_five_and_nonpositive_cash():
    data, _, _, _ = fixture()
    source = cycle_sources(data)
    score, state = efficiency_score(source[MODELS[0]].iloc[:5])
    assert state == 'COMPLETE_SCORE' and score['return_sum'] == pytest.approx(.06)
    assert score['absolute_return_sum'] == pytest.approx(.10) and score['score'] == pytest.approx(.6)
    row, selected = cycle_budget_at(source, data.date.iloc[32], (.5, .5))
    assert row['budget_reference'] == pytest.approx(.375) and row['budget_runs'] == pytest.approx(.625)
    insufficient, _ = cycle_budget_at(source, data.date.iloc[25], (.3, .7))
    assert insufficient['status'] == 'INSUFFICIENT_MATURE_CYCLES' and insufficient['budget_reference'] == .3
    row, selected = cycle_budget_at(source, data.date.iloc[-1], (.5, .5))
    assert row['reference_count'] == row['runs_count'] == 20
    assert [r['cycle'] for r in selected if r['source_model'] == MODELS[0]] == list(range(11, 31))
    source[MODELS[0]]['cycle_net_return'] = -.01
    source[MODELS[1]]['cycle_net_return'] = 0.
    cash, _ = cycle_budget_at(source, data.date.iloc[-1], (.5, .5))
    assert cash['status'] == 'CASH_NONPOSITIVE_EFFICIENCIES' and cash['budget_reference'] == cash['budget_runs'] == 0.
    source[MODELS[0]].loc[29, 'complete_information'] = False
    unavailable, _ = cycle_budget_at(source, data.date.iloc[-1], (.3, .7))
    assert unavailable['status'] == 'INCOMPLETE_SELECTED_CYCLE' and unavailable['budget_reference'] == .3


def test_month_first_updates_same_budget_separate_period_and_maturity_before_selection():
    data, parents, _, start = fixture()
    source = cycle_sources(data)
    frames, summary, events, selections = runs_closed_cycle_budget_frames(data, parents, source, CFG, start)
    f = frames['BASE']
    assert events[0]['status'] == 'INITIAL_HALF' and events[0]['budget_reference'] == .5
    assert summary[0]['positive_efficiency_updates'] > 0
    for t in range(3, len(data)-1):
        if data.date.iloc[t].month == data.date.iloc[t-1].month:
            assert f.budget_reference.iloc[t] == f.budget_reference.iloc[t-1]
    assert all(r['maturity_date'] <= r['origin'] for r in selections)
    pd.testing.assert_series_equal(f.budget_reference, frames['STRESS'].budget_reference)
    source[MODELS[0]].loc[29, 'maturity_date'] = data.date.iloc[-1]+pd.Timedelta(days=1)
    row, selected = cycle_budget_at(source, data.date.iloc[-1], (.5, .5))
    assert [r['cycle'] for r in selected if r['source_model'] == MODELS[0]] == list(range(10, 30))
    later_data, later_parents, _, later_start = fixture(first=220)
    later_source = cycle_sources(later_data, first=220, count=20)
    reset, _, later_events, _ = runs_closed_cycle_budget_frames(later_data, later_parents, later_source, CFG, later_start)
    assert later_events[0]['reference_count'] == 0 and reset['BASE'].budget_reference.iloc[219] == .5


def test_future_cycles_terminal_prefix_and_known_clock():
    data, parents, _, start = fixture()
    source = cycle_sources(data, count=50)
    original, _, _, _ = runs_closed_cycle_budget_frames(data, parents, source, CFG, start)
    changed = {model: f.copy() for model, f in source.items()}
    for f in changed.values():
        f.loc[f.exit_date.ge(data.date.iloc[250]), 'cycle_net_return'] = -.5
    future, _, _, _ = runs_closed_cycle_budget_frames(data, parents, changed, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:250], future['BASE'].iloc[:250])
    for f in changed.values():
        f.loc[f.index[-1], ['exit_date', 'maturity_date']] = data.date.iloc[-1]
    terminal_row, selected = cycle_budget_at(changed, data.date.iloc[-2], (.5, .5))
    assert not any(r['cycle'] == 50 for r in selected)
    prefix_parents = {cost: {model: f[f.origin_index.lt(329)].copy() for model, f in group.items()} for cost, group in parents.items()}
    prefix_sources = {model: f[f.exit_date.lt(data.date.iloc[329])].copy() for model, f in source.items()}
    short, _, _, _ = runs_closed_cycle_budget_frames(data.iloc[:330], prefix_parents, prefix_sources, CFG, start)
    pd.testing.assert_frame_equal(original['BASE'].iloc[:329], short['BASE'].iloc[:329])
    assert pd.DatetimeIndex(original['BASE'].decision_time.iloc[2:-1]).equals(pd.DatetimeIndex(data.date.iloc[2:-1])+pd.Timedelta(hours=15, minutes=5))
    assert original['BASE'][PRIMARY+'_target'].iloc[:2].isna().all() and pd.isna(original['BASE'][PRIMARY+'_target'].iloc[-1])


def test_zero_budget_parent_unknown_and_source_fee_identity():
    data, parents, _, start = fixture()
    source = cycle_sources(data, values=[np.full(30, -.01), np.full(30, .02)])
    frames, _, events, _ = runs_closed_cycle_budget_frames(data, parents, source, CFG, start)
    t = next(r['origin_index'] for r in events if r['status'] == 'UPDATED_POSITIVE_EFFICIENCY')
    assert frames['BASE'][PRIMARY+'_target'].iloc[t] == .2 and frames['STRESS'][PRIMARY+'_target'].iloc[t] == .1
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(t), 'reference_weight'] = np.nan
    unknown, _, _, _ = runs_closed_cycle_budget_frames(data, parents, source, CFG, start)
    assert pd.isna(unknown['BASE'][PRIMARY+'_target'].iloc[t])
    source[MODELS[0]]['cost'] = 'STRESS'
    with pytest.raises(ValueError):
        runs_closed_cycle_budget_frames(data, parents, source, CFG, start)


def test_real_funds_cycle_budget_cash_exit_reentry_unknown_dividend_and_terminal():
    account_boundaries()
    original, parents, _, start = fixture(n=600)
    raw = original.close.to_numpy().copy(); raw[9:] -= .1
    div = np.zeros(len(raw)); div[9] = .1
    data = prices(raw, div); data.loc[9, 'open'] = raw[8]-.1
    returns = np.r_[np.full(5, .05), np.full(25, -.01), np.full(30, .01)]
    source = cycle_sources(data, count=60, values=[returns, returns])
    parents['BASE'][MODELS[0]].loc[parents['BASE'][MODELS[0]].origin_index.eq(8), 'reference_weight'] = np.nan
    frames, _, events, _ = runs_closed_cycle_budget_frames(data, parents, source, CFG, start)
    target = frames['BASE'][PRIMARY+'_target'].to_numpy(float)
    assert any(r['status'] == 'CASH_NONPOSITIVE_EFFICIENCIES' for r in events)
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
