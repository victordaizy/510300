"""只用成熟的实际持仓周期，按过去盈亏效率每月分配预算。"""
import json
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = 'RUNS_CLOSED_CYCLE_BUDGET'
MODELS = ['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE']
CANDIDATES = {PRIMARY: '两父已成熟周期盈亏效率的月度预算'}


def mature_saved_cycles(data, ledger, saved, dividends, model, first):
    """核对保存周期的买入支出、净利润和归属分红，再标明最早可读日。"""
    dates = pd.DatetimeIndex(data.date)
    require(ledger.source_model.eq(model).all() and ledger.source_cost.eq('BASE').all(), '周期预算父账户身份或基础费用不同')
    require(saved.source_model.eq(model).all() and saved.cost.eq('BASE').all(), '保存周期来源或费用不同')
    require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), '周期父账户与完整日历不同')
    require(ledger.mark_clock.iloc[:-1].eq('CLOSE').all() and ledger.mark_clock.iloc[-1] == 'OPEN_TERMINAL', '周期父账户终点时钟不同')
    cycles = saved.copy().sort_values('entry_date').reset_index(drop=True)
    for column in ['entry_date', 'exit_date']:
        cycles[column] = pd.to_datetime(cycles[column])
    require(not cycles.cycle.duplicated().any() and cycles.cycle.to_list() == list(range(1, len(cycles)+1)), '保存周期编号或顺序不同')
    actual_entries = ledger.loc[ledger.filled_quantity.gt(0) & ledger.shares_before.eq(0), 'date']
    actual_exits = ledger.loc[ledger.filled_quantity.lt(0) & ledger.shares.eq(0), 'date']
    require(pd.DatetimeIndex(cycles.entry_date).equals(pd.DatetimeIndex(actual_entries)) and
        pd.DatetimeIndex(cycles.exit_date).equals(pd.DatetimeIndex(actual_exits)), '保存周期不对应实际首次买入和全部退出')
    account = ledger.set_index('date')
    fields = ['buy_debit', 'gross_price_profit', 'commission', 'slippage', 'dividend_recognized', 'net_profit', 'cycle_net_return']
    records = []
    for row in cycles.itertuples():
        period_rows = ledger[ledger.date.between(row.entry_date, row.exit_date)]
        require(row.entry_date < row.exit_date and period_rows.shares.iloc[:-1].gt(0).all(), '周期没有连续实际持仓或未遵守次日可卖')
        buys = period_rows[period_rows.filled_quantity.gt(0)]
        debit = float((buys.filled_quantity*buys.fill_price+buys.commission).sum())
        gross = float((-period_rows.filled_quantity*period_rows.open_price).sum())
        commission = float(period_rows.commission.sum())
        slippage = float(period_rows.slippage_cost.sum())
        maturity = row.exit_date
        recognized = 0.
        event_count = 0
        for event in dividends.itertuples():
            if event.record_date in account.index and row.entry_date <= event.record_date < row.exit_date:
                shares = int(account.loc[event.record_date, 'shares'])
                if shares > 0:
                    require(pd.notna(event.ex_date) and np.isfinite(event.cash_dividend_per_share), '归属分红缺少明确除息时点或金额')
                    event_count += 1
                    maturity = max(maturity, event.ex_date)
                    if event.ex_date in account.index:
                        recognized += shares*event.cash_dividend_per_share
        expected = {'buy_debit': debit, 'gross_price_profit': gross, 'commission': commission,
            'slippage': slippage, 'dividend_recognized': recognized,
            'net_profit': gross+recognized-commission-slippage}
        require(debit > 0, '完整周期没有正的累计实际买入支出')
        expected['cycle_net_return'] = expected['net_profit']/debit
        complete = True
        for field in fields:
            value = float(getattr(row, field))
            require(np.isfinite(value) or np.isnan(value), '保存周期含非法无穷数')
            if np.isnan(value):
                complete = False
            else:
                require(abs(value-expected[field]) <= (1e-10 if field == 'cycle_net_return' else 1e-6), '保存周期资金、费用或分红不能还原')
        require(bool(row.terminal_exit) == (row.exit_date == dates[-1]), '保存周期终点退出标记不同')
        records.append({'source_model': model, 'cost': 'BASE', 'cycle': int(row.cycle),
            'entry_date': row.entry_date, 'exit_date': row.exit_date, 'maturity_date': maturity,
            'dividend_events': event_count, 'complete_information': complete,
            'buy_debit': float(row.buy_debit), 'net_profit': float(row.net_profit),
            'cycle_net_return': float(row.cycle_net_return) if complete else np.nan,
            'terminal_exit': bool(row.terminal_exit)})
    return pd.DataFrame(records, columns=['source_model', 'cost', 'cycle', 'entry_date', 'exit_date', 'maturity_date',
        'dividend_events', 'complete_information', 'buy_debit', 'net_profit', 'cycle_net_return', 'terminal_exit'])


def efficiency_score(selected):
    values = selected.cycle_net_return.to_numpy(float)
    row = {'count': len(selected), 'return_sum': np.nan, 'absolute_return_sum': np.nan, 'score': np.nan,
        'win_cycles': 0, 'loss_cycles': 0, 'flat_cycles': 0}
    if len(selected) < 5:
        return row, 'INSUFFICIENT_MATURE_CYCLES'
    if not selected.complete_information.all() or not np.isfinite(values).all():
        return row, 'INCOMPLETE_SELECTED_CYCLE'
    total, absolute = float(values.sum()), float(np.abs(values).sum())
    row.update(return_sum=total, absolute_return_sum=absolute, score=max(0., total/absolute) if absolute > 0 else 0.,
        win_cycles=int((values > 0).sum()), loss_cycles=int((values < 0).sum()), flat_cycles=int((values == 0).sum()))
    return row, 'COMPLETE_SCORE'


def cycle_budget_at(source_cycles, origin, previous):
    require(set(source_cycles) == set(MODELS), '成熟周期父来源集合不同')
    require(len(previous) == 2 and all(np.isfinite(v) and 0 <= v <= 1 for v in previous)
        and (sum(previous) == 0 or abs(sum(previous)-1.) < 1e-12), '上一成熟周期预算非法')
    row = {'old_budget_reference': previous[0], 'old_budget_runs': previous[1],
        'budget_reference': previous[0], 'budget_runs': previous[1]}
    states, scores, selections = [], [], []
    for label, model in zip(['reference', 'runs'], MODELS):
        source = source_cycles[model]
        require(source.source_model.eq(model).all() and source.cost.eq('BASE').all(), '成熟周期身份或费用不同')
        selected = source[source.exit_date.le(origin) & source.maturity_date.le(origin)].sort_values('exit_date').tail(20)
        diagnostics, state = efficiency_score(selected)
        row.update({label+'_'+key: value for key, value in diagnostics.items()})
        row[label+'_cycle_ids'] = json.dumps(selected.cycle.astype(int).to_list())
        states.append(state)
        scores.append(diagnostics['score'])
        for cycle in selected.itertuples():
            selections.append({'origin': origin, 'source_model': model, 'cycle': int(cycle.cycle),
                'exit_date': cycle.exit_date, 'maturity_date': cycle.maturity_date,
                'cycle_net_return': cycle.cycle_net_return, 'complete_information': bool(cycle.complete_information)})
    if 'INCOMPLETE_SELECTED_CYCLE' in states:
        row['status'] = 'INCOMPLETE_SELECTED_CYCLE'
    elif 'INSUFFICIENT_MATURE_CYCLES' in states:
        row['status'] = 'INSUFFICIENT_MATURE_CYCLES'
    else:
        total = sum(scores)
        row.update(budget_reference=scores[0]/total if total > 0 else 0., budget_runs=scores[1]/total if total > 0 else 0.,
            status='UPDATED_POSITIVE_EFFICIENCY' if total > 0 else 'CASH_NONPOSITIVE_EFFICIENCIES')
    return row, selections


def runs_closed_cycle_budget_frames(data, parents_by_cost, source_cycles, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['parent_models'] == MODELS and
        cfg['cycle_window'] == 20 and cfg['minimum_mature_cycles'] == 5 and cfg['initial_budgets'] == [.5, .5] and
        cfg['budget_update'] == 'FIRST_TRADING_DAY_MONTH_CLOSE' and cfg['budget_source_cost'] == 'BASE',
        '成熟周期盈亏效率的固定预算规则改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    dates = pd.DatetimeIndex(data.date)
    origins = np.arange(first-1, len(data)-1)
    budgets = np.full((len(data), 2), np.nan)
    statuses = np.full(len(data), 'OUTSIDE_DECISION', dtype=object)
    monthly = np.zeros(len(data), bool)
    previous = (.5, .5)
    events, selections = [], []
    for t in origins:
        if t == first-1:
            row, chosen = cycle_budget_at(source_cycles, dates[t], previous)
            require(row['reference_count'] == row['runs_count'] == 0, '评价准备收盘已有被借用的历史周期')
            row['status'] = 'INITIAL_HALF'
        elif dates[t].to_period('M') != dates[t-1].to_period('M'):
            monthly[t] = True
            row, chosen = cycle_budget_at(source_cycles, dates[t], previous)
            previous = row['budget_reference'], row['budget_runs']
        else:
            budgets[t] = previous
            statuses[t] = 'MONTHLY_BUDGET_HELD'
            continue
        events.append({'origin_index': int(t), 'origin': dates[t], **row})
        selections.extend(chosen)
        budgets[t] = previous
        statuses[t] = row['status']
    summaries = []
    for cost, frame in frames.items():
        a = frame[MODELS[0]+'_parent_target'].to_numpy(float)
        b = frame[MODELS[1]+'_parent_target'].to_numpy(float)
        target = budgets[:, 0]*a+budgets[:, 1]*b
        frame['budget_reference'], frame['budget_runs'] = budgets[:, 0], budgets[:, 1]
        frame['budget_status'], frame['month_first_update_attempt'] = statuses, monthly
        frame['budget_source_cost'] = 'BASE'
        frame[PRIMARY+'_target'] = target
        current = target[origins]
        summaries.append({'model': PRIMARY, 'cost': cost, 'decision_origins': len(origins),
            'positive_target_origins': int((current > 0).sum()), 'zero_target_origins': int((current == 0).sum()),
            'unknown_target_origins': int(np.isnan(current).sum()), 'month_first_attempts': int(monthly.sum()),
            'positive_efficiency_updates': sum(row['status'] == 'UPDATED_POSITIVE_EFFICIENCY' for row in events),
            'nonpositive_cash_updates': sum(row['status'] == 'CASH_NONPOSITIVE_EFFICIENCIES' for row in events),
            'mean_reference_budget': float(budgets[origins, 0].mean()), 'mean_runs_budget': float(budgets[origins, 1].mean()),
            'cash_budget_origins': int((budgets[origins].sum(axis=1) == 0).sum()),
            'mean_target': float(np.nanmean(current)) if np.isfinite(current).any() else None})
    return frames, summaries, events, selections
