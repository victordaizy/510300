"""固定510300已选策略的经济诊断；全部写入独立目录。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_strategy_review_diagnostics_v1'
MAIN = ROOT / 'reports/research/510300_september_monthly_continuation_v1'
EARLY = ROOT / 'reports/research/510300_incremental_selected_intent_mix_v1'
INPUTS = ROOT / 'reports/research/510300_post_selection_extension_inputs_v1'
MODEL = 'SELECTED_MIX_BAND10_SIMPLE2'
FEATURES = ['log_holding_days', 'cycle_return', 'cycle_drawdown', 'entry_mode', 'mom5', 'mom20', 'sma120', 'vol20']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(x) for x in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat() if pd.notna(value) else None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name, value):
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    path = OUT / (name + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return frame


def inputs():
    cfg = read(ROOT / 'config/510300_incremental_selected_intent_mix_v1.json')
    cfg['data_cutoff'] = '2026-09-11'
    data = pd.read_parquet(INPUTS / 'candidate_features.parquet')
    data.date = pd.to_datetime(data.date)
    from research.adaptive_allocation_v1 import normalize_dividends
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    return cfg, data, dividends


def saved(period, cost, model=MODEL, folder=None):
    if period == 'main':
        path = MAIN / 'accounts' / cost / model
        return pd.read_parquet(path / 'ledger.parquet'), pd.read_parquet(path / 'decisions.parquet')
    path = (folder or EARLY) / 'earlier_diagnostic' / cost
    return pd.read_parquet(path / f'{model}_ledger.parquet'), pd.read_parquet(path / f'{model}_decisions.parquet')


def metrics(ledger, capital=200000.):
    returns = ledger.net_return.to_numpy(float)
    nav = np.r_[capital, ledger.equity.to_numpy(float)]
    np.testing.assert_allclose(returns, nav[1:] / nav[:-1] - 1, atol=1e-12, rtol=0)
    vol = float(np.std(returns, ddof=1) * np.sqrt(242))
    return {'days': len(ledger), 'end_equity': float(nav[-1]), 'profit': float(nav[-1]-capital),
            'annual_return': float((nav[-1]/capital)**(242/len(ledger))-1),
            'sharpe': float(np.mean(returns)*242/vol) if vol else None,
            'volatility': vol, 'max_drawdown': float(np.min(nav/np.maximum.accumulate(nav)-1)),
            'fills': int(ledger.filled_quantity.ne(0).sum()), 'commission': float(ledger.commission.sum()),
            'slippage': float(ledger.slippage_cost.sum()), 'mean_exposure': float(ledger.exposure.mean()),
            'max_accounting_error': float(ledger.accounting_error.abs().max())}


def cycles(ledger, capital=200000.):
    rows, active, previous = [], None, capital
    for i, row in enumerate(ledger.itertuples()):
        if row.filled_quantity > 0 and row.shares_before == 0:
            assert active is None
            active = {'entry': row.date, 'start_index': i, 'start_equity': previous}
        if active is not None and row.shares == 0:
            piece = ledger.iloc[active['start_index']:i+1]
            nav = np.r_[active['start_equity'], piece.equity]
            active.update(exit=row.date, end_index=i, days=len(piece), profit=row.equity-active['start_equity'],
                          total_return=row.equity/active['start_equity']-1, end_equity=row.equity,
                          drawdown=float(np.min(nav/np.maximum.accumulate(nav)-1)),
                          commission=piece.commission.sum(), slippage=piece.slippage_cost.sum())
            rows.append(active)
            active = None
        previous = row.equity
    return pd.DataFrame(rows), active


def save_run(name, cost, run, capital=200000., period='main'):
    ledger, decisions, state = run[0], run[1], run[-1]
    folder = OUT / 'counterfactuals' / name / period / cost
    folder.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(folder / 'ledger.parquet', index=False)
    decisions.to_parquet(folder / 'decisions.parquet', index=False)
    write(folder / 'checkpoint.json', state)
    m = {'scenario': name, 'period': period, 'cost': cost, **metrics(ledger, capital)}
    write(folder / 'metrics.json', m)
    return m


def freeze():
    if (OUT / 'protocol_freeze.json').exists():
        raise RuntimeError('本轮协议已冻结，直接运行未完成的诊断部分。')
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), ROOT / 'research/strategy_review_full_graph_v1.py', ROOT / 'docs/510300_STRATEGY_REVIEW_DIAGNOSTICS_V1.md',
             ROOT / 'config/510300_research_authority_v6.json',
             ROOT / 'config/510300_incremental_selected_intent_mix_v1.json',
             MAIN / 'result.json', MAIN / 'within_models.json', MAIN / 'ridge_models.json',
             INPUTS / 'candidate_features.parquet', INPUTS / 'dependency_graph.json',
             ROOT / 'data/reference/510300_dividends.csv',
             ROOT / 'reports/research/510300_fixed_research_origin_v1/origin.json',
             ROOT / 'reports/research/510300_fixed_date_continuation_validation_v1/saved_verification_receipt.json']
    paths += list((MAIN / 'accounts').glob('*/*/*.parquet'))
    paths += list((MAIN / 'factors').glob('*.parquet'))
    paths += list((MAIN / 'training_reference').glob('*'))
    paths += [ROOT / n['configuration'] for n in read(INPUTS / 'dependency_graph.json')['nodes']]
    paths = sorted(set(p for p in paths if p.is_file()))
    write(OUT / 'protocol_freeze.json', {'frozen_at': datetime.now().astimezone().isoformat(),
          'status': 'FROZEN_BEFORE_NEW_COUNTERFACTUALS', 'model': MODEL,
          'historical_only': True, 'new_model_fits': 0, 'position_impact': 0,
          'seed': 20260913, 'paired_cycle_repetitions': 5000,
          'inputs': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in paths]})
    print('本轮诊断协议和直接输入已经冻结。', flush=True)


def basic():
    cfg, data, dividends = inputs()
    market = data.set_index('date')
    rows, all_cycles, risks, entries, execution, yearly = [], [], [], [], [], []
    focus = []
    for period in ['main', 'earlier']:
        for cost in cfg['costs']:
            ledger, decisions = saved(period, cost)
            m = metrics(ledger)
            cs, unfinished = cycles(ledger)
            assert unfinished is None
            assert abs(cs.profit.sum()-m['profit']) < 1e-6
            ranked = cs.sort_values('profit', ascending=False)
            m.update(period=period, cost=cost, cycles=len(cs), top1_fraction=ranked.profit.iloc[0]/m['profit'],
                     top5_fraction=ranked.profit.head(5).sum()/m['profit'])
            rows.append(m)
            all_cycles.append(cs.assign(period=period, cost=cost))
            held = ledger.shares.gt(0)
            high = ledger.exposure.gt(.8)
            risk = {'period': period, 'cost': cost, 'cash_close_days': int((~held).sum()),
                    'cash_close_fraction': float((~held).mean()), 'held_close_days': int(held.sum()),
                    'mean_exposure_held': float(ledger.loc[held, 'exposure'].mean()),
                    'maximum_exposure': float(ledger.exposure.max()), 'above80_days': int(high.sum()),
                    'worst_high_exposure_day_return': float(ledger.loc[ledger.exposure.shift(1).gt(.8), 'net_return'].min()),
                    'maximum_target': float(decisions.reference_weight.max())}
            risks.append(risk)
            dec = decisions.set_index('execution_date')
            for row in ledger[ledger.filled_quantity.gt(0) & ledger.shares_before.eq(0)].itertuples():
                d = dec.loc[row.date]
                vol = float(market.loc[d.origin, 'vol20'])
                entries.append({'period': period, 'cost': cost, 'entry': row.date, 'origin': d.origin,
                                'target': d.reference_weight, 'known_etf_vol20': vol,
                                'target_times_volatility_proxy': d.reference_weight*vol,
                                'entry_open_exposure': row.shares*row.open/(row.cash+row.shares*row.open+row.dividend_receivable)})
            for y, group in ledger.groupby(ledger.date.dt.year):
                previous = 200000. if group.index[0] == 0 else float(ledger.equity.iloc[group.index[0]-1])
                yearly.append({'period': period, 'cost': cost, 'year': int(y),
                               'actual_return': float(group.equity.iloc[-1]/previous-1),
                               **metrics(group, previous)})
            if period == 'main':
                focus.extend(ranked.head(5).assign(cost=cost).to_dict('records'))
                from research.intraday_overnight_increment_v1 import fill_price, commission
                for row in ranked.head(5).itertuples():
                    exit_trade = ledger.iloc[row.end_index]
                    quantity = abs(int(exit_trade.filled_quantity))
                    index = int(np.flatnonzero(data.date.eq(row.exit))[0])
                    prices = {'ORIGINAL_OPEN': data.open.iloc[index], 'SAME_DAY_CLOSE': data.close.iloc[index]}
                    if index+1 < len(data):
                        prices['NEXT_DAY_OPEN'] = data.open.iloc[index+1]
                    original_net = quantity*exit_trade.fill_price-exit_trade.commission
                    for name, price in prices.items():
                        fill = fill_price(float(price), -1, cfg['costs'][cost], cfg['tick'])
                        net = quantity*fill-commission(quantity, fill, cfg['costs'][cost])
                        execution.append({'cost': cost, 'entry': row.entry, 'exit': row.exit, 'scenario': name,
                                          'quantity': quantity, 'raw_price': price, 'fill_price': fill,
                                          'fixed_quantity_exit_cash_delta': net-original_net,
                                          'cycle_profit_if_only_exit_cash_changed': row.profit+net-original_net,
                                          'status': 'FIXED_QUANTITY_PRICE_SENSITIVITY_NOT_EXECUTION_EVIDENCE'})
    table('account_metrics', rows)
    table('all_complete_cycles', pd.concat(all_cycles, ignore_index=True))
    table('holding_risk', risks)
    table('entry_risk', entries)
    table('annual_results', yearly)
    table('opening_price_sensitivity', execution)
    table('top5_cycles', focus)
    trace_dec, trace_led = [], []
    targets = pd.read_parquet(MAIN / 'factors/all_required_targets.parquet')
    focus_dates = set()
    for f in focus:
        a = int(np.flatnonzero(data.date.eq(f['entry']))[0])-2
        b = int(np.flatnonzero(data.date.eq(f['exit']))[0])+1
        focus_dates.update(data.date.iloc[max(a, 0):b+1])
    for cost in cfg['costs']:
        for folder in (MAIN / 'accounts' / cost).iterdir():
            for name, destination in [('decisions', trace_dec), ('ledger', trace_led)]:
                d = pd.read_parquet(folder / (name+'.parquet'))
                key = 'origin' if name == 'decisions' else 'date'
                destination.append(d[d[key].isin(focus_dates)].assign(node=folder.name, cost=cost))
    table('top5_all_node_decisions', pd.concat(trace_dec, ignore_index=True))
    table('top5_all_node_ledgers', pd.concat(trace_led, ignore_index=True))
    table('top5_all_targets', targets[targets.date.isin(focus_dates)])
    model_rows, freshness = [], []
    for kind in ['ridge', 'within']:
        models = read(MAIN / (kind+'_models.json'))['models']
        for m in models:
            freshness.append({k: m.get(k) for k in ['fit_origin', 'status', 'training_cycle_count', 'training_rows', 'latest_exit_date']} | {'kind': kind})
        if kind == 'within':
            for cost in cfg['costs']:
                _, ds = saved('main', cost, 'ENTRY_VINTAGE_EXIT')
                for row in ds[ds.continuation_prediction.notna()].to_dict('records'):
                    m = next(x for x in models if pd.Timestamp(x['fit_origin']) == row['learning_fit_origin'])
                    obj = m['model']
                    z = np.clip((np.array([row[k] for k in FEATURES])-np.array(obj['mean']))/np.array(obj['scale']), -obj['feature_clip'], obj['feature_clip'])
                    contrib = z*np.array(obj['coefficients'])
                    pred = float(obj['intercept']+contrib.sum())
                    assert abs(pred-row['continuation_prediction']) < 1e-12
                    model_rows.append({'cost': cost, 'origin': row['origin'], 'cycle_id': row['learning_cycle_id'],
                                       'model_fit_origin': m['fit_origin'], 'prediction': pred, 'intercept': obj['intercept'],
                                       **dict(zip([k+'_contribution' for k in FEATURES], contrib)),
                                       'without_cycle_return_contribution': pred-contrib[1],
                                       'exit_requested': row['learned_exit_requested']})
    table('model_freshness', freshness)
    table('within_model_prediction_contributions', model_rows)
    samples = pd.read_parquet(MAIN / 'training_reference/samples.parquet')
    centered = []
    for cycle, group in samples.groupby('cycle_id'):
        if len(group) < 3:
            continue
        x = group.cycle_return.to_numpy(float); y = group.target.to_numpy(float)
        dx, dy = x-x.mean(), y-y.mean()
        centered.append({'cycle_id': cycle, 'rows': len(group), 'mature_date': group.mature_date.max(),
                         'correlation': float(np.corrcoef(dx, dy)[0, 1]) if np.std(dx)*np.std(dy) else None,
                         'centered_slope': float(dx@dy/(dx@dx)) if dx@dx else None})
    table('label_within_cycle_relationship', centered)
    # 与评审相同的近似式只是描述标签结构；实际标签含下一开盘分母、价位及分红。
    write(OUT / 'basic_receipt.json', {'status': 'PASS_SAVED_FOUR_ACCOUNT_RECOMPUTATION_AND_TRACE',
          'metrics': rows, 'simulation_fills': sum(m['fills'] for m in rows), 'new_models': 0,
          'minute_vwap_or_capacity': 'BLOCKED_NO_HISTORICAL_MINUTE_AUCTION_ORDER_EVIDENCE',
          'strict_forward_days': 0, 'completed_at': datetime.now().astimezone().isoformat()})
    print('四账户复算、前五周期全链条、模型贡献和风险表已保存。', flush=True)


def exit_pairs():
    from research.post_selection_continuous_accounts_v1 import simulate_rearmed_exit
    from research.simple_intraday_protection_v1 import make_rules
    from research.learned_cycle_exit_v1 import ExitController
    from research.within_cycle_exit_inputs_v1 import WithinCycleExitController
    from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
    cfg, data, div = inputs()
    within = read(MAIN / 'within_models.json')['models']
    ridge = read(MAIN / 'ridge_models.json')['models']
    specification = read(ROOT / 'config/510300_entry_vintage_exit_v1.json')['specification']
    rule = make_rules(data)['D60_INTRA']
    natural = pd.read_csv(MAIN / 'training_reference/cycles.csv')
    natural['entry_date'] = pd.to_datetime(natural.entry_date, format='mixed')
    natural['exit_date'] = pd.to_datetime(natural.exit_date, format='mixed')
    completed = natural[natural.exit_date.notna() & natural.entry_date.ge('2015-01-05')].copy()
    variants = ['PRICE_ORIGINAL', 'RIDGE_MONTHLY', 'WITHIN_MONTHLY', 'WITHIN_ENTRY_FIXED', 'TIME20', 'PROFIT08']
    rows, ledgers, decisions = [], [], []
    for ci, cycle in enumerate(completed.itertuples()):
        entry_idx = int(cycle.entry_index)
        end_idx = int(np.flatnonzero(data.date.eq(cycle.exit_date))[0])
        mask = np.zeros(len(data), dtype=int); mask[entry_idx-1] = int(cycle.mode)
        isolated = {**rule, 'entry': mask}
        record = within[bisect_right([x['fit_index'] for x in within], entry_idx)-1]
        mature = record['status'] == 'FIT_COMPLETE'
        for cost in cfg['costs']:
            first_quantity = None
            for name in variants:
                spec = deepcopy(specification)
                controller = None
                if name == 'RIDGE_MONTHLY':
                    controller = ExitController(data, ridge, 2)
                elif name == 'WITHIN_MONTHLY':
                    controller = WithinCycleExitController(data, within, 2)
                elif name == 'WITHIN_ENTRY_FIXED':
                    controller = EntryVintageExitController(data, within, 2)
                elif name == 'TIME20':
                    spec['modes']['1']['days'] = 20
                elif name == 'PROFIT08':
                    spec['modes']['1']['take'] = .08
                run = simulate_rearmed_exit(data, div, cfg, cfg['costs'][cost], str(cycle.entry_date.date()),
                                           isolated, spec, controller, stop_index=end_idx, terminal_liquidation=True)
                ledger, ds = run[:2]
                buys = ledger[ledger.filled_quantity.gt(0)]
                assert len(buys) == 1 and buys.date.iloc[0] == cycle.entry_date
                quantity = int(buys.filled_quantity.iloc[0])
                first_quantity = quantity if first_quantity is None else first_quantity
                assert quantity == first_quantity
                assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-6
                sells = ledger[ledger.filled_quantity.lt(0)]
                row = {'cycle_id': int(cycle.cycle_id), 'entry': cycle.entry_date, 'natural_exit': cycle.exit_date,
                       'cost': cost, 'variant': name, 'mature_model_at_entry': mature,
                       'entry_model_fit': record['fit_origin'], 'entry_quantity': quantity,
                       'actual_exit': sells.date.iloc[-1], 'cycle_return': float(ledger.equity.iloc[-1]/200000.-1),
                       'profit': float(ledger.equity.iloc[-1]-200000.),
                       'entry_period': 'main' if cycle.entry_date >= pd.Timestamp('2020-01-02') else 'earlier',
                       'exit_reason': sells.execution_reasons.iloc[-1]}
                rows.append(row)
                ledgers.append(ledger.assign(pair_cycle_id=int(cycle.cycle_id), variant=name, cost=cost))
                decisions.append(ds.assign(pair_cycle_id=int(cycle.cycle_id), variant=name, cost=cost))
        print(f'固定入场退出对照 {ci+1}/{len(completed)} 个自然完整周期完成。', flush=True)
    frame = table('paired_exit_cycles', rows)
    pd.concat(ledgers, ignore_index=True).to_parquet(OUT / 'paired_exit_ledgers.parquet', index=False)
    pd.concat(decisions, ignore_index=True).to_parquet(OUT / 'paired_exit_decisions.parquet', index=False)
    table('unfinished_reference_cycles', natural[natural.exit_date.isna()])
    summary = []
    rng = np.random.default_rng(20260913)
    for cost in cfg['costs']:
        for scope in ['all_mature', 'main_mature', 'earlier_mature']:
            subset = frame[(frame.cost == cost) & frame.mature_model_at_entry]
            if scope != 'all_mature':
                subset = subset[subset.entry_period.eq(scope.split('_')[0])]
            pivot = subset.pivot(index='cycle_id', columns='variant', values='cycle_return')
            ids = rng.integers(0, len(pivot), (5000, len(pivot)))
            np.savez_compressed(OUT / f'paired_indices_{cost}_{scope}.npz', indices=ids, cycles=pivot.index.to_numpy())
            for name in variants:
                delta = (pivot[name]-pivot.PRICE_ORIGINAL).to_numpy(float)
                boot = delta[ids].mean(axis=1)
                summary.append({'cost': cost, 'scope': scope, 'variant': name, 'complete_cycles': len(delta),
                                'mean_cycle_return': float(pivot[name].mean()), 'mean_increment': float(delta.mean()),
                                'median_increment': float(np.median(delta)), 'better_cycles': int((delta>1e-12).sum()),
                                'worse_cycles': int((delta < -1e-12).sum()), 'same_cycles': int((np.abs(delta)<=1e-12).sum()),
                                'delta_q025': float(np.quantile(boot, .025)), 'delta_q975': float(np.quantile(boot, .975))})
    table('paired_exit_summary', summary)
    write(OUT / 'exit_pairs_receipt.json', {'status': 'PASS_FIXED_ENTRY_COMPLETE_CYCLE_PAIRED_DIAGNOSTIC',
          'completed_cycles': len(completed), 'isolated_accounts': len(frame), 'new_model_fits': 0,
          'same_entry_quantity_within_cost': True, 'full_final_strategy_replacement': False,
          'independent_validation': False, 'unfinished_cycles': int(natural.exit_date.isna().sum())})
    print('固定入场的六种退出对照与周期级配对区间已完成。', flush=True)


def run_single(name, cost, targets, cfg, data, div, period='main', policy=None, start=None, capital=None, band=.1):
    from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
    from research.adaptive_allocation_v1 import target_request
    settings = {**cfg, 'weight_band': band, 'initial_capital': capital or cfg['initial_capital']}
    destination = OUT / 'counterfactuals' / name / period / cost
    if (destination / 'metrics.json').exists():
        return (pd.read_parquet(destination / 'ledger.parquet'), pd.read_parquet(destination / 'decisions.parquet'),
                read(destination / 'checkpoint.json'))
    if policy is None:
        policy = lambda a, p, v, c, m, t: target_request(a, p, v, c)
    result = simulate_indexed_request_account(data, div, settings, settings['costs'][cost],
                start or ('2020-01-02' if period == 'main' else '2015-01-05'), name, targets=targets,
                event_mask=np.ones(len(data), bool), request_policy=policy,
                terminal_liquidation=period == 'earlier', next_execution_date='2026-09-14')
    save_run(name, cost, result, settings['initial_capital'], period)
    return result


def capped_internal_targets(raw, multiplier60, multiplier30):
    """只限制正目标的风险放大，保持明确零与未知的原始语义。"""
    from research.episode_account_risk_budget_inputs_v1 import fixed_episode_budget
    raw = np.asarray(raw, float)
    daily = np.clip(raw*np.minimum(1., np.asarray(multiplier60, float)), 0, 1)
    daily[raw == 0] = 0.
    episode, _, _ = fixed_episode_budget(raw, np.minimum(1., np.asarray(multiplier30, float)))
    return daily, episode


def account_probes():
    from research.post_selection_continuous_factors_v1 import planned_weights
    from research.adaptive_allocation_v1 import target_request
    from research.two_close_zero_exit_inputs_v1 import zero_streak, confirmation_request
    from research.addition_gate_exposure_batch_inputs_v1 import gate_request, market_factors
    cfg, data, div = inputs()
    first = int(np.flatnonzero(data.date.ge('2020-01-02'))[0])
    all_targets = pd.read_parquet(MAIN / 'factors/all_required_targets.parquet')
    result_rows = []
    for period in ['main', 'earlier']:
        frame = data if period == 'main' else data[data.date.le('2019-12-31')].reset_index(drop=True)
        base, base_dec = saved(period, 'BASE')
        targets = np.full(len(frame), np.nan)
        targets[base_dec.origin_index.astype(int)] = base_dec.reference_weight
        replay = run_single('BASE_TARGET_REPLAY_CHECK', 'BASE', targets, cfg, frame, div, period)
        for column in ['equity', 'cash', 'shares', 'net_return', 'filled_quantity', 'requested_quantity']:
            np.testing.assert_allclose(replay[0][column], base[column], atol=1e-7, rtol=0, equal_nan=True)
        # 固定数量请求不假定可执行；原引擎保留现金、可卖份额和涨跌停约束。
        requests = dict(zip(base_dec.origin_index.astype(int), base_dec.requested_quantity.astype(int)))
        def fixed_request(a, p, v, c, m, t):
            return {'requested_quantity': requests[t], 'reference_weight': v,
                    'action': '基础历史申请数量固定，按当前资金和压力成本尝试执行'}
        run_single('BASE_REQUESTS_STRESS_COST', 'STRESS', targets, cfg, frame, div, period, policy=fixed_request)
        run_single('BASE_TARGETS_STRESS_COST', 'STRESS', targets, cfg, frame, div, period)
        for cost in cfg['costs']:
            for name, values in [('BUY_HOLD', np.ones(len(frame))), ('CONSTANT20', np.full(len(frame), .2)),
                                 ('ETF_VOL10', np.minimum(1., .1/frame.vol20.to_numpy(float)))]:
                run_single(name, cost, values, cfg, frame, div, period)
    for cost in cfg['costs']:
        original, original_dec = saved('main', cost)
        main_target = all_targets[MODEL+'__'+cost].to_numpy(float)
        for name, values, band in [('FINAL_RISK_CAP10', np.minimum(main_target, np.minimum(1., .1/data.vol20)), .1),
                                  ('NO_FINAL_BAND', main_target, 0.)]:
            run_single(name, cost, np.asarray(values), cfg, data, div, band=band)
        for source, name in [('ADD_GATE_EXPOSURE_115', 'SOURCE_A_ONLY'), ('RUNS_OPPORTUNITY_CAPPED_SUM', 'SOURCE_B_ONLY')]:
            ledger, ds = saved('main', cost, source)
            weights = planned_weights(data, ds, ledger, cfg, first)
            run_single(name, cost, weights, cfg, data, div)
        risk = pd.read_parquet(MAIN / f'factors/account_risk_{cost}.parquet')
        raw = risk.source_target.to_numpy(float)
        for variant in ['NO_INTERNAL_UPSCALE', 'NO_ZERO_WAIT', 'NO_115_MULTIPLIER']:
            local = {}
            if variant == 'NO_INTERNAL_UPSCALE':
                daily, episode = capped_internal_targets(raw, risk.multiplier60, risk.multiplier30)
            else:
                daily = all_targets['ACCOUNT_VOLATILITY_EXPOSURE__'+cost].to_numpy(float)
                episode = all_targets['RISK_EPISODE_30_10__'+cost].to_numpy(float)
            for key, values in [('TWO_CLOSE_ZERO_EXIT', daily), ('CONFIRMED_RISK_EPISODE_30_10', episode)]:
                counts = zero_streak(values)
                settings = {**cfg, 'zero_confirmations': 2, 'weight_band': .1}
                def policy(a, p, v, c, m, t, counts=counts, variant=variant):
                    if variant == 'NO_ZERO_WAIT':
                        return target_request(a, p, v, c)
                    return confirmation_request(a, p, v, c, 'TWO_CLOSE_ZERO_EXIT', int(counts[t]))
                local[key] = run_single(variant+'__'+key, cost, values, settings, data, div, policy=policy)
            rebound = np.minimum(1., daily+.5*all_targets['R2_Z_CONFIRM__'+cost].to_numpy(float))
            local['MEAN_REBOUND_AUX_50'] = run_single(variant+'__MEAN_REBOUND_AUX_50', cost, rebound, cfg, data, div)
            combined = sum(w*planned_weights(data, run[1], run[0], cfg, first) for w, run in zip([.8, .15, .05], local.values()))
            exposure = np.minimum(1., combined*(1. if variant == 'NO_115_MULTIPLIER' else 1.15))
            market = market_factors(data)
            source_cfg = read(ROOT / 'config/510300_addition_gate_exposure_batch_v1.json')
            def addition(a, p, v, c, m, t):
                return gate_request(a, p, v, c, 'ADD_GATE_EXPOSURE_115', bool(market.buy_allowed_lower_r01.iloc[t]), float(market.daily_total_simple.iloc[t]))
            aa = run_single(variant+'__SOURCE_A', cost, exposure, source_cfg, data, div, policy=addition)
            bb_l, bb_d = saved('main', cost, 'RUNS_OPPORTUNITY_CAPPED_SUM')
            master = np.minimum(1., .85*planned_weights(data, aa[1], aa[0], cfg, first)+.15*planned_weights(data, bb_d, bb_l, cfg, first))
            run_single(variant, cost, master, cfg, data, div)
            print(f'模块诊断完成：{variant}／{cost}。', flush=True)
    # 在真实持仓日分割新最终账户，验证完整状态恢复。
    from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
    base_l, base_d = saved('main', 'BASE')
    split = int(np.flatnonzero(data.date.eq('2024-09-27'))[0])
    values = all_targets[MODEL+'__BASE'].to_numpy(float)
    kwargs = dict(targets=values, event_mask=np.ones(len(data), bool),
                  request_policy=lambda a, p, v, c, m, t: target_request(a, p, v, c), next_execution_date='2026-09-14')
    args = (data, div, {**cfg, 'weight_band': .1}, cfg['costs']['BASE'], '2020-01-02', 'RESUME_DIAGNOSTIC')
    a = simulate_indexed_request_account(*args, **kwargs, stop_index=split)
    write(OUT / 'held_resume_checkpoint.json', a[-1])
    b = simulate_indexed_request_account(*args, **kwargs, resume=read(OUT / 'held_resume_checkpoint.json'))
    joined = pd.concat([a[0], b[0]], ignore_index=True)
    for key in ['equity', 'shares', 'cash', 'net_return', 'filled_quantity']:
        np.testing.assert_allclose(joined[key], base_l[key], atol=1e-7, rtol=0)
    write(OUT / 'held_resume_receipt.json', {'status': 'PASS_HELD_FINAL_ACCOUNT_SERIALIZED_RESUME', 'split': '2024-09-27',
          'shares_at_split': int(a[0].shares.iloc[-1]), 'full_graph_resume_evidence': 'REUSED_ROUND215_EXACT_SPLIT_RECEIPT'})
    for path in (OUT / 'counterfactuals').glob('*/*/*/metrics.json'):
        result_rows.append(read(path))
    table('counterfactual_account_metrics', result_rows)
    write(OUT / 'account_probes_receipt.json', {'status': 'PASS_NEW_FIXED_COUNTERFACTUAL_ACCOUNTS', 'accounts': len(result_rows),
          'new_model_fits': 0, 'historical_only': True, 'baseline_reconstruction_checks': 2})
    print('模块、成本、简单配置及持仓恢复诊断完成。', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='510300固定策略诊断')
    parser.add_argument('part', choices=['freeze', 'basic', 'exits', 'accounts'])
    part = parser.parse_args().part
    if part != 'freeze' and not (OUT / 'protocol_freeze.json').exists():
        raise RuntimeError('请先冻结诊断协议。')
    started = time.perf_counter()
    {'freeze': freeze, 'basic': basic, 'exits': exit_pairs, 'accounts': account_probes}[part]()
    write(OUT / ('part_'+part+'_receipt.json'), {'part': part, 'status': 'COMPLETED', 'seconds': time.perf_counter()-started,
          'completed_at': datetime.now().astimezone().isoformat(), 'program_sha256': digest(Path(__file__))})


if __name__ == '__main__':
    main()
