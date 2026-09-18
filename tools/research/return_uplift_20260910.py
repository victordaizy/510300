"""用户授权的固定增收研究：原账户复算、参考替换、一次资金倍率对照。

只读取已经保存的510300历史资料，不下载市场数据、不拟合模型、不产生实盘信号。
运行：python tools/research/return_uplift_20260910.py
"""
from __future__ import annotations
import json
import math
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.event_clock_account_v1 import simulate_event_account
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.simple_intraday_protection_v1 import make_rules
from research.simple_signal_blend_v1 import decision_state
from research.two_policy_min_variance_inputs_v1 import budget_frame

OUT = ROOT / 'research_runs/return_uplift_20260910'
P = ROOT / 'reports/research'
FOLDERS = {
    'R91': '510300_continuous_reference_min_variance_v1',
    'R128': '510300_entry_vintage_exit_v1',
    'R143': '510300_trend_noise_reference_blend_v1',
    'R150': '510300_monotone_episode_budget_v1',
}
MODELS = {
    'R91': 'CONTINUOUS_REFERENCE_MIN_VARIANCE',
    'R128': 'ENTRY_VINTAGE_EXIT',
    'R143': 'TREND_NOISE_REFERENCE_BLEND',
    'R150': 'EPISODE_BUDGET_NONINCREASING',
}
NEW_MODELS = ['REPLACE_EXIT_KEEP_91_BUDGET', 'REPLACE_EXIT_REESTIMATE_242_BUDGET', 'R143_FIXED_2X_CAPPED', 'R150_FIXED_2X_CAPPED']
NAMES = {
    'REPLACE_EXIT_KEEP_91_BUDGET': '以128连续参考替换旧退出，保留91预算',
    'REPLACE_EXIT_REESTIMATE_242_BUDGET': '替换退出后重新估计原242日最小方差预算',
    'R143_FIXED_2X_CAPPED': '143目标固定乘二，最高100%',
    'R150_FIXED_2X_CAPPED': '150目标固定乘二，最高100%',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def cfg_for(key):
    return read_json(ROOT / 'config' / (FOLDERS[key] + '.json'))


def read_parquet(path):
    """读取副本统一时间精度，避免pandas版本造成ms/ns日历比较误报。"""
    frame = pd.read_parquet(path)
    for col in frame.columns:
        if getattr(frame[col].dtype, "kind", None) == "M":
            frame[col] = frame[col].astype("datetime64[ns]")
    return frame


def stored(key, period, cost, suffix='ledger'):
    return read_parquet(P / FOLDERS[key] / period / cost / f'{MODELS[key]}_{suffix}.parquet')


def aligned(data, decisions):
    """只用判断时点意向，不将下一开盘成交倒填为信号。"""
    require(not decisions.origin.duplicated().any(), '保存意向出现重复日期')
    ids = pd.DatetimeIndex(data.date).get_indexer(pd.to_datetime(decisions.origin))
    require((ids >= 0).all() and (ids < len(data)-1).all(), '保存意向索引越界')
    require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(data.date.iloc[ids+1])), '意向不是下一开盘执行')
    return decision_state(data, decisions)


def capped_double(x):
    x = np.asarray(x, float)
    require((np.isnan(x) | ((x >= 0) & (x <= 1))).all(), '原目标范围异常')
    return np.minimum(1., 2. * x)


def check_replay(key, period, cost, actual, expected):
    require(pd.DatetimeIndex(actual.date).equals(pd.DatetimeIndex(expected.date)), '复算日历不同')
    diffs = {}
    for col in ['equity', 'cash', 'shares', 'filled_quantity', 'commission', 'slippage_cost', 'net_return']:
        diff = float(np.abs(actual[col].to_numpy(float)-expected[col].to_numpy(float)).max())
        diffs[col] = diff
        require(diff < (1e-12 if col == 'net_return' else 1e-6), f'{key}/{period}/{cost}/{col}复算不一致:{diff}')
    return {'model': key, 'period': period, 'cost': cost, 'rows': len(actual), 'status': 'PASS', 'max_abs_differences': diffs}


def save_path(folder, model, ledger, decisions):
    folder.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(folder / f'{model}_ledger.parquet', index=False)
    decisions.to_parquet(folder / f'{model}_decisions.parquet', index=False)
    ledger.to_csv(folder / f'{model}_ledger.csv.gz', index=False)
    decisions.to_csv(folder / f'{model}_decisions.csv.gz', index=False)


def cycle_summary(ledger):
    rows, active, start, total, held = [], False, None, 0., 0
    prev = 0
    for r in ledger.itertuples():
        if not active and r.shares > 0:
            active, start, total, held = True, r.date, 0., 0
        if active:
            total += r.pnl
            held += int(r.shares > 0)
        if active and r.shares == 0 and prev > 0:
            rows.append({'entry_date': start, 'exit_date': r.date, 'pnl': total, 'holding_closes': held})
            active, start, total, held = False, None, 0., 0
        prev = r.shares
    require(not active, '研究终点没有清算')
    values = np.array([r['pnl'] for r in rows])
    net = float(ledger.pnl.sum())
    positive = np.sort(values[values > 0])[::-1]
    holding_days = int(ledger.shares.gt(0).sum())
    return {'cycles': len(rows), 'losing_cycles': int((values < 0).sum()), 'net_profit': net,
        'holding_days': holding_days, 'holding_day_share': holding_days/len(ledger),
        'conditional_exposure_when_held': float(ledger.loc[ledger.shares.gt(0), 'exposure'].mean()) if holding_days else None,
        'max_exposure': float(ledger.exposure.max()), 'top_three_profit_share': float(positive[:3].sum()/net) if net else None,
        'top_five_profit_share': float(positive[:5].sum()/net) if net else None,
        'pnl_outside_closed_cycles': net-float(values.sum())}, pd.DataFrame(rows)


def boot_pair(a, b, annual_days, rng, repetitions=2000, block=20):
    """固定路径的联合循环区块区间，不是独立样本验证或全研究家族校正。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    require(a.shape == b.shape and (a > -1).all() and (b > -1).all(), '比较路径无效')
    values = []
    for _ in range(math.ceil(repetitions/100)):
        starts = rng.integers(0, n, size=(100, math.ceil(n/block)))
        ids = ((starts[:, :, None]+np.arange(block)) % n).reshape(100, -1)[:, :n]
        x, y = a[ids], b[ids]
        ca = np.expm1(np.log1p(x).sum(axis=1)*annual_days/n)
        cb = np.expm1(np.log1p(y).sum(axis=1)*annual_days/n)
        values.extend((ca-cb).tolist())
    values = np.asarray(values[:repetitions])
    point = float(np.expm1(np.log1p(a).sum()*annual_days/n)-np.expm1(np.log1p(b).sum()*annual_days/n))
    return {'cagr_difference': point, 'ci_025': float(np.quantile(values, .025)), 'ci_975': float(np.quantile(values, .975)),
            'bootstrap_positive_fraction_descriptive_only': float((values > 0).mean())}


def inventory():
    rows = []
    docs = list(P.glob('*/result.json'))
    for path in docs:
        obj = read_json(path)
        cp = ROOT / 'config' / (path.parent.name + '.json')
        config = read_json(cp) if cp.exists() else {}
        primary = obj.get('primary', [])
        if not isinstance(primary, list):
            continue
        early = obj.get('earlier_diagnostics', [])
        if not isinstance(early, list):
            early = []
        for row in primary:
            if not isinstance(row, dict) or row.get('cost') != 'BASE':
                continue
            mid = row.get('model')
            all4 = [r for r in primary+early if isinstance(r, dict) and r.get('model') == mid]
            valid = [r.get('net_sharpe') for r in all4 if r.get('net_sharpe') is not None]
            rows.append({'round': config.get('round', 0), 'folder': path.parent.name, 'model': mid,
                'main_base_sharpe': row.get('net_sharpe'), 'main_base_cagr': row.get('annualized_return'),
                'four_scenario_minimum_sharpe': min(valid) if len(valid) == 4 else None})
    frame = pd.DataFrame(rows).sort_values('round')
    frame.to_csv(OUT / 'existing_primary_inventory.csv', index=False)
    return {'result_documents_scanned': len(docs), 'primary_metric_rows': len(frame), 'last_saved_round': int(frame['round'].max()),
            'after_round_130': frame[frame['round'].gt(130)].to_dict('records')}


def main():
    began = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / 'protocol.json').exists(), '本轮已登记；禁止覆盖或根据结果重调')
    config = {k: cfg_for(k) for k in FOLDERS}
    cfg = config['R91']
    for other in config.values():
        for k in ['evaluation_start','data_cutoff','initial_capital','lot','tick','annual_days','cash_annual_rate_assumption','costs','earlier_start','earlier_terminal','weight_band']:
            if k in other:
                require(cfg[k] == other[k], '比较口径不一致:'+k)
    synthetic = np.array([np.nan, 0., .1, .5, .8, 1.])
    np.testing.assert_equal(capped_double(synthetic), [np.nan, 0., .2, 1., 1., 1.])
    # 读取源码与保存输入的摘要；本次不得改变任何旧冻结材料。
    used = [ROOT / 'input_export_manifest.json', Path(__file__)]
    if (ROOT / 'later_input_manifest.json').exists(): used.append(ROOT / 'later_input_manifest.json')
    used.extend(ROOT / 'config' / (f+'.json') for f in FOLDERS.values())
    used.extend(ROOT.glob('research/*.py'))
    used.append(ROOT / cfg['features'].replace('\\','/'))
    used.append(ROOT / cfg['dividends'].replace('\\','/'))
    for f in FOLDERS.values():
        used.extend(p for p in (P/f).rglob('*') if p.is_file() and p.suffix in {'.parquet','.json'})
    used.extend((P/'510300_within_cycle_exit_v1').glob('saved_models.json'))
    frozen = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(used))]
    protocol = {'study_id': '510300_RETURN_UPLIFT_FIXED_DIAGNOSTIC_20260910', 'registered_at': now(),
        'source_commit': 'e5433a24e6aecc2e3333ac12e050fbb84f9169ce', 'scope': ['510300','CASH_CNY'],
        'new_candidates': NEW_MODELS, 'planned_new_accounts': 16, 'planned_original_account_replays': 16,
        'new_fits': 0, 'parameter_search': False, 'candidate_selection_uses_previously_observed_history': True,
        'reference_cost_for_B_C': 'BASE_FOR_BOTH_COSTS_MATCHING_ORIGINAL_91',
        'multiplier_D_E': 2., 'position_cap': 1., 'risk_window': 242, 'weight_band': cfg['weight_band'],
        'capital': cfg['initial_capital'], 'costs': cfg['costs'], 'annual_days': cfg['annual_days'],
        'cash_return_and_risk_free_rate': 0., 'bootstrap_block': 20, 'bootstrap_repetitions': 2000, 'seed': 20260910,
        'rule_B': 'Use 91 saved point-in-time budgets; replace only old learned-reference intent with continuous 128-reference intent.',
        'rule_C': 'Same panic reference and new 128 reference; original 242-day monthly min-variance budgets with continuous 2013 history.',
        'rule_D_E': 'Double respectively saved 143 and 150 targets once, cap at 100%; preserve zeros and unknowns; same-cost source intentions.',
        'stop': 'No signal/window/threshold/direction/multiplier rescue; all four candidates and scenarios reported regardless of outcome.',
        'independent_validation': 'NOT_ESTABLISHED', 'live_trading_authorized': False, 'position_impact': 0, 'frozen_files': frozen}
    write_json(OUT/'protocol.json', protocol, exclusive=True)
    print('PROTOCOL_FROZEN', digest(OUT/'protocol.json'), flush=True)
    inv = inventory()
    data = read_parquet(ROOT/cfg['features'].replace('\\','/'))
    div = normalize_dividends(pd.read_csv(ROOT/cfg['dividends'].replace('\\','/')))
    models = read_json(ROOT/config['R128']['saved_models'].replace('\\','/'))['models']
    require(len(models) == 141, '模型记录数量变化')
    for record in models:
        if record['status'] == 'FIT_COMPLETE':
            require(record['latest_exit_index'] <= record['fit_index'], '保存训练记录存在未来周期')
    replays, metrics, coverage, all_accounts, attribution = [], [], [], {}, []
    periods = [('evaluation', data, cfg['evaluation_start']),
               ('earlier_diagnostic', data[data.date.le(cfg['earlier_terminal'])].copy(), cfg['earlier_start'])]
    # 先复算16个旧账户，不通过即停止，不计算任何新候选账户。
    for period, frame, start in periods:
        for cost_id, cost in cfg['costs'].items():
            for key in ['R91','R143','R150']:
                expected = stored(key, period, cost_id)
                dec = stored(key, period, cost_id, 'decisions')
                target = aligned(frame, dec)
                actual, _ = simulate_event_account(frame, div, config[key], cost, start, key, targets=target, event_mask=np.ones(len(frame), bool))
                replays.append(check_replay(key, period, cost_id, actual, expected))
                all_accounts[(period,cost_id,key)] = expected
            expected = stored('R128', period, cost_id)
            actual, _, _ = simulate_rearmed_exit(frame, div, config['R128'], cost, start, make_rules(frame)['D60_INTRA'],
                config['R128']['specification'], EntryVintageExitController(frame, models, config['R128']['confirmation_days']))
            replays.append(check_replay('R128', period, cost_id, actual, expected))
            all_accounts[(period,cost_id,'R128')] = expected
            all_accounts[(period,cost_id,'BUY_HOLD')] = read_parquet(P/FOLDERS['R150']/period/cost_id/'BUY_HOLD_ledger.parquet')
    write_json(OUT/'replay_checks.json', {'status':'PASS', 'checks':replays})
    print('ORIGINAL_ACCOUNT_REPLAYS_PASS', len(replays), flush=True)
    new_ref, new_dec, ref_cycles = simulate_rearmed_exit(data, div, config['R128'], cfg['costs']['BASE'], cfg['reference_start'],
        make_rules(data)['D60_INTRA'], config['R128']['specification'], EntryVintageExitController(data, models, config['R128']['confirmation_days']))
    save_path(OUT/'continuous_reference', 'VINTAGE_128_BASE', new_ref, new_dec)
    ref_cycles.to_csv(OUT/'continuous_reference/cycles.csv', index=False)
    panic_folder = P/FOLDERS['R91']/'continuous_references/BASE'
    panic_ledger = read_parquet(panic_folder/'PANIC_ONLY_ledger.parquet')
    panic_dec = read_parquet(panic_folder/'PANIC_ONLY_decisions.parquet')
    first_ref = int(np.flatnonzero(data.date.ge(cfg['reference_start']))[0])
    ref_returns = np.full((len(data),2), np.nan)
    for col, ledger in enumerate([panic_ledger,new_ref]):
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first_ref:])), '连续参考日历不一致')
        ref_returns[first_ref:,col] = ledger.net_return.to_numpy(float)
    states = np.column_stack([aligned(data,panic_dec), aligned(data,new_dec)])
    new_budget = budget_frame(data.date, ref_returns, states, first_ref, 242)
    # 固定前缀测试：删去2020年及以后数据，不改变此前的参考状态或预算。
    early_frame = periods[1][1]
    pre_ref, pre_dec, _ = simulate_rearmed_exit(early_frame, div, config['R128'], cfg['costs']['BASE'], cfg['reference_start'],
        make_rules(early_frame)['D60_INTRA'], config['R128']['specification'], EntryVintageExitController(early_frame,models,config['R128']['confirmation_days']))
    np.testing.assert_allclose(pre_ref.equity.iloc[:-1], new_ref.equity.iloc[:len(pre_ref)-1], atol=1e-6, rtol=0)
    np.testing.assert_array_equal(pre_dec.reference_weight.to_numpy(), new_dec.reference_weight.iloc[:len(pre_dec)].to_numpy())
    pre_budget = budget_frame(early_frame.date, ref_returns[:len(early_frame)], states[:len(early_frame)], first_ref, 242)
    np.testing.assert_allclose(pre_budget.target.iloc[:-1],new_budget.target.iloc[:len(early_frame)-1],equal_nan=True,atol=1e-12,rtol=0)
    write_json(OUT/'temporal_checks.json', {'continuous_reference_prefix':'PASS','budget_prefix':'PASS','saved_model_maturity':'PASS','new_fits':0})
    for period, frame, start in periods:
        old_factors = read_parquet(P/FOLDERS['R91']/f'{period}_factors.parquet')
        require(pd.DatetimeIndex(old_factors.date).equals(pd.DatetimeIndex(frame.date)), '原预算日历不匹配')
        b_target = old_factors.panic_budget.to_numpy(float)*states[:len(frame),0] + old_factors.learned_budget.to_numpy(float)*states[:len(frame),1]
        c_target = new_budget.target.iloc[:len(frame)].to_numpy(float).copy()
        b_target[-1], c_target[-1] = np.nan, np.nan
        for cost_id, cost in cfg['costs'].items():
            t143=aligned(frame,stored('R143',period,cost_id,'decisions'))
            t150=aligned(frame,stored('R150',period,cost_id,'decisions'))
            targets=dict(zip(NEW_MODELS,[b_target,c_target,capped_double(t143),capped_double(t150)]))
            first=int(np.flatnonzero(frame.date.ge(start))[0])
            for mid, target in targets.items():
                eligible=target[first-1:-1]
                require(np.isfinite(eligible).all() and (eligible>=0).all() and (eligible<=1+1e-12).all(), '候选有未知或越界目标')
                ledger, decisions=simulate_event_account(frame,div,cfg,cost,start,mid,targets=target,event_mask=np.ones(len(frame),bool))
                require(ledger.accounting_error.abs().max()<1e-6 and not ledger.terminal_unliquidated.iloc[-1], '新账户记账或结算失败')
                require((ledger.cash>=-1e-6).all() and (ledger.shares>=0).all(), '新账户透支或做空')
                all_accounts[(period,cost_id,mid)]=ledger
                save_path(OUT/period/cost_id,mid,ledger,decisions)
                pd.DataFrame({'date':frame.date,'target':target}).to_csv(OUT/period/cost_id/f'{mid}_targets.csv.gz',index=False)
                coverage.append({'period':period,'cost':cost_id,'model':mid,'target_origins':len(eligible),'positive_target_origins':int((eligible>0).sum()),
                    'targets_at_cap':int((eligible>=1-1e-12).sum()),'max_target':float(eligible.max()),'mean_target':float(eligible.mean())})
            print('NEW_ACCOUNTS_COMPLETE',period,cost_id,flush=True)
    cycles_all=[]
    for (period,cost_id,mid),ledger in all_accounts.items():
        bench=all_accounts[(period,cost_id,'BUY_HOLD')]
        m={'period':period,'cost':cost_id,'model':mid,**summarize(ledger,cfg)}
        m['cagr_excess_vs_buy_hold']=m['annualized_return']-summarize(bench,cfg)['annualized_return']
        cs, cy=cycle_summary(ledger)
        m.update(cs)
        metrics.append(m)
        if not cy.empty:
            cycles_all.append(cy.assign(period=period,cost=cost_id,model=mid))
    mf=pd.DataFrame(metrics)
    mf.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'target_coverage.csv',index=False)
    pd.concat(cycles_all,ignore_index=True).to_csv(OUT/'actual_cycles.csv',index=False)
    # 旧32到128的路径差异：按共同/新增/减少风险日期归因，保留原有账户复利。
    for period,frame,start in periods:
        for cost_id in cfg['costs']:
            old=read_parquet(P/'510300_rearmed_session_exit_v1'/period/cost_id/'REARM_RIDGE_ledger.parquet')
            new=all_accounts[(period,cost_id,'R128')]
            require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(new.date)), '32/128归因日历不匹配')
            a=old.shares.gt(0)|old.shares.shift(fill_value=0).gt(0)
            b=new.shares.gt(0)|new.shares.shift(fill_value=0).gt(0)
            cat=np.select([a&b,~a&b,a&~b],['BOTH_EXPOSED','ONLY_128_EXPOSED','ONLY_32_EXPOSED'],default='BOTH_CASH')
            delta=np.log1p(new.net_return)-np.log1p(old.net_return)
            for name in np.unique(cat):
                mask=cat==name
                attribution.append({'period':period,'cost':cost_id,'category':name,'days':int(mask.sum()),'log_wealth_ratio_contribution':float(delta[mask].sum())})
    pd.DataFrame(attribution).to_csv(OUT/'exit_path_attribution.csv',index=False)
    boot=[]
    for period,_,_ in periods:
        for cost_id in cfg['costs']:
            for a,b in [(NEW_MODELS[0],'R91'),(NEW_MODELS[1],'R91'),(NEW_MODELS[1],NEW_MODELS[0]),(NEW_MODELS[2],'R143'),(NEW_MODELS[3],'R150')]:
                stat=boot_pair(all_accounts[(period,cost_id,a)].net_return,all_accounts[(period,cost_id,b)].net_return,cfg['annual_days'],np.random.default_rng(20260910))
                boot.append({'period':period,'cost':cost_id,'candidate':a,'control':b,**stat})
    pd.DataFrame(boot).to_csv(OUT/'paired_block_intervals.csv',index=False)
    for row in frozen:
        require(digest(ROOT/row['path'])==row['sha256'],'运行期间冻结来源改变:'+row['path'])
    summary={'study_id':protocol['study_id'],'completed_at':now(),'status':'COMPLETED_FIXED_HISTORICAL_RESEARCH_ONLY',
        'source_commit':protocol['source_commit'],'protocol_sha256':digest(OUT/'protocol.json'),'runtime_python':platform.python_version(),
        'runtime_numpy':np.__version__,'runtime_pandas':pd.__version__,'runtime_seconds':time.perf_counter()-began,
        'source_scan':inv,'original_replays_passed':len(replays),'new_candidate_accounts':len(NEW_MODELS)*4,'new_model_fits':0,
        'new_continuous_reference':1,'prefix_test_reference_replay':1,'original_files_unchanged':True,
        'all_four_sharpe_1_2_candidates':[m for m in NEW_MODELS if mf[mf.model.eq(m)].net_sharpe.ge(1.2).all()],
        'all_four_positive_excess_candidates':[m for m in NEW_MODELS if mf[mf.model.eq(m)].cagr_excess_vs_buy_hold.gt(0).all()],
        'independent_validation':'NOT_ESTABLISHED','goal_achieved':False,'position_impact':0,'live_trading_authorized':False,
        'metrics':metrics,'paired_block_intervals':boot,'exit_path_attribution':attribution}
    write_json(OUT/'result.json',summary,exclusive=True)
    print(mf[['period','cost','model','annualized_return','net_sharpe','max_drawdown','mean_exposure','trade_count']].to_string(index=False),flush=True)
    print('COMPLETE',digest(OUT/'result.json'),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write_json(OUT/'execution_failure.json',{'at':now(),'type':type(exc).__name__,'error':str(exc),'goal_achieved':False,'position_impact':0})
        raise
