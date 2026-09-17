"""两套来源买卖意图合并账户的一次计算和保存核对。"""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.three_source_order_intent_mix_inputs_v1 import CANDIDATES, MODELS, PRIMARY, ROOT, SOURCES, intent_frames
from scripts.verify_round195_20260913 import prices_and_costs

OUT = ROOT/'reports/research/510300_three_source_order_intent_mix_v1'
CONFIG = ROOT/'config/510300_three_source_order_intent_mix_v1.json'
MIX = ROOT/'reports/research/510300_incremental_saved_mix_through199/result.json'
PARENTS = {m: ROOT/'reports/research'/folder for m, folder in SOURCES.items()}
CONTROLS = {
    'TWO_CLOSE_ZERO_EXIT': (PARENTS[MODELS[0]], '原198两次归零确认'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '原181完整账户波动预算'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本轮已经登记或启动')
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    tested = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_three_source_order_intent_mix_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr, encoding='utf-8')
    print(tested.stdout, flush=True)
    require(tested.returncode == 0 and '5 passed' in tested.stdout, '五项合并规则测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': tested.returncode, 'passed': 5,
        'seconds': time.perf_counter()-began}, exclusive=True)
    old_path = ROOT/'config/510300_two_close_zero_exit_v1.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal', 'weight_band']
    cfg = {k: old[k] for k in keys}
    mix = json.loads(MIX.read_text(encoding='utf-8'))
    require([r['model'] for r in mix['best_weights']] == MODELS, '已保存前三来源身份不同')
    values = np.array([r['weight'] for r in mix['best_weights']])
    values /= values.sum()
    cfg.update(study_id='510300_THREE_SOURCE_ORDER_INTENT_MIX_V1', round=200, registered_at=now(),
        primary=PRIMARY, candidate_models=list(CANDIDATES), candidate_configurations=2,
        parent_models=MODELS, source_folders=SOURCES, source_weights={PRIMARY: dict(zip(MODELS, [.8, .15, .05])),
        'INTENT_MIX_DIAGNOSTIC_WEIGHTS': dict(zip(MODELS, values.tolist()))},
        decision_clock='15:05:00', annual_return_target=.10, new_model_fits=0, new_reference_accounts=0,
        source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_THREE_SOURCE_PLANNED_EXPOSURE_NETTING',
        rules='docs/510300_THREE_SOURCE_ORDER_INTENT_MIX_V1.md')
    text = (ROOT/'docs/510300_THREE_SOURCE_ORDER_INTENT_MIX_NEXT_20260913.md').read_text(encoding='utf-8')
    text = text.replace('尚未实现、测试、冻结或计算本轮账户。', '本轮五项必要测试通过后、首次新账户计算前确定以下规则。')
    text += '\n\n## 两套实际冻结权重\n\n|来源|简单权重|已保存筛选权重|\n|---|---:|---:|\n'
    for m, label in zip(MODELS, ['两次归零确认', '三十日区间固定预算与两次归零确认', '二分之一均值反弹辅助']):
        text += f"|{label}|{cfg['source_weights'][PRIMARY][m]:.2%}|{cfg['source_weights']['INTENT_MIX_DIAGNOSTIC_WEIGHTS'][m]:.12%}|\n"
    text += '\n\n## 三个来源的预算和反弹因素\n\n'
    text += ('共同基础是第174轮的任一方向确认辅助目标，下面附其完整中文规则。第181轮取该参考账户截至当日收盘最近六十个完整日净收益的样本标准差，乘二百四十二的平方根；'
        '风险倍率为10%除以该年化波动，乘第174轮目标后封顶100%。评价起点后的样本不足六十日或完整零波动，倍率用一；足量但缺失时倍率未知。原目标明确零仍为零。基础和压力各用相应费用参考账户的收益。\n\n'
        '来源一在第181轮目标上使用两次明确零确认：第一次零保持实有份额，连续第二次零才请求下一开盘全退。正值或未知打断计数，未知保持份额。正目标按原十个百分点调仓带向目标中心申请；没有永久退出锁，恢复正目标可以取消等待。\n\n'
        '来源二取同一第174轮账户最近三十日净收益波动，风险预算仍为10%。仅在原174目标由非正状态进入连续正目标区间时确定倍率，整个正目标区间沿用；原174目标变零结束区间，未知不结束。原目标比例在区间内仍可变化。该缩放目标也按与来源一相同的两次明确零确认退出。\n\n'
        '来源三把第181轮目标与二分之一的第23轮反弹参考目标相加，封顶100%，任一来源未知则合成未知。反弹参考用含分红简单日收益连乘生成累计财富，计算当日财富相对最近二十日均值的偏离，再除以同窗口样本标准差。'
        '原保存特征在标准差为零或无法计算时将偏离记为零，但仍要求原必要输入完整；不将新缺失行情补成有效信号。空仓且数据完整、该偏离分数严格小于负一点五、当日财富严格高于前一日，且退出后已等待至少一个交易日，下一开盘尝试买入。'
        '持有后任一条件成立则请求退出：偏离分数回到零或以上、本笔含分红持仓价值相对实际买入金额与佣金亏损达到5%、或含买入日在内已持有十个收盘交易日。待退出状态持续至实际卖出；未知不能伪造新信号。反弹参考正目标为100%，实际进入按现金与费用可负担的最多整百份，受次日可卖约束。'
        '来源三合成账户按十个百分点调仓带执行，合成零立即退出，终点开盘清算。\n\n'
        '以上三个已保存参考账户只用于生成收盘计划；新合并账户不使用其下一开盘成交和收益，也不将参考费用再次扣除。下文保留共同174来源的全部因素与原参考规则；其中旧轮次的计算状态及结论只说明来源，不代表第200轮状态。\n\n')
    common = ROOT/'docs/510300_RETURN_CONFIRMATION_AUXILIARY_BATCH_V1.md'
    text += common.read_text(encoding='utf-8')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        stream.write(text)
    paths = [Path(__file__), ROOT/'research/three_source_order_intent_mix_inputs_v1.py',
        ROOT/'tests/test_three_source_order_intent_mix_v1.py', ROOT/'research/saved_target_batch_runner_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py', ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/event_clock_account_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'scripts/verify_round195_20260913.py', old_path, common, MIX, OUT/'tests_output.txt', OUT/'tests_receipt.json']
    paths += [ROOT/cfg[k] for k in ['features', 'dividends', 'rules']]
    for model, folder in PARENTS.items():
        paths += [folder/'saved_verification_receipt.json', ROOT/'config'/f'{SOURCES[model]}.json']
        for period in ['evaluation', 'earlier_diagnostic']:
            for cost in cfg['costs']:
                paths += [folder/period/cost/f'{model}_{kind}.parquet' for kind in ['ledger', 'decisions']]
    for model, (folder, _) in CONTROLS.items():
        paths += [folder/p/c/f'{model}_ledger.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(paths))]
    index_path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 199 and not index['running_studies'], '前序完成状态不同')
    write_json(CONFIG, cfg, exclusive=True)
    index['running_studies'] = [{'round': 200, 'study': cfg['study_id'], 'status': 'FROZEN_NOT_STARTED', 'config': str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='THREE_SOURCE_ORDER_INTENT_MIX_FROZEN', source=cfg['rules'])
    write_json(index_path, index)
    print('第200轮两套合并方案已冻结，计划八条完整账户。', flush=True)


def run():
    with (OUT/'run_output.txt').open('x', encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
        result = run_saved_target_batch(ROOT, OUT, CONFIG, CANDIDATES, PARENTS, CONTROLS, intent_frames)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    joint = save_joint_assessment(OUT, cfg, result)
    print(json.dumps({'核心计算秒数': result['run_seconds'], '联合验收': joint['candidates'],
        '新账户结果': [{**r, 'period': p} for p, k in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]
                      for r in result[k] if r['model'] in CANDIDATES]}, ensure_ascii=False), flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本轮核对已经完成')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cache, checks = {}, []

    def expected(period, cost, frame, start, candidate):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        folder = OUT/period/cost
        if (period, cost) not in cache:
            factors = pd.read_parquet(folder/'factors.parquet')
            plans = {}
            for model, old in PARENTS.items():
                source = pd.read_parquet(old/period/cost/f'{model}_decisions.parquet')
                ledger = pd.read_parquet(old/period/cost/f'{model}_ledger.parquet')
                np.testing.assert_array_equal(source.origin_index, indices)
                require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源收盘日期不同')
                require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(ledger.date)), '来源下一开盘与账本不同')
                require(pd.DatetimeIndex(source.decision_time).equals(pd.DatetimeIndex(source.origin)+pd.Timedelta(hours=15, minutes=5)), '来源决定时钟不同')
                values = []
                for i, row in enumerate(source.itertuples()):
                    equity = cfg['initial_capital'] if i == 0 else float(ledger.equity.iloc[i-1])
                    shares = 0 if i == 0 else int(ledger.shares.iloc[i-1])
                    planned = shares+int(row.requested_quantity)
                    price = float(frame.close.iloc[indices[i]])
                    value = planned*price/equity
                    require(0 <= value <= 1+1e-12 and planned >= 0, '来源独立计划越界')
                    values.append(np.nan if pd.isna(row.reference_weight) else min(1., value))
                plans[model] = np.asarray(values)
                np.testing.assert_allclose(factors.loc[indices, model+'_planned_weight'], values, atol=1e-12, rtol=0, equal_nan=True)
            for control, (old, _) in CONTROLS.items():
                pd.testing.assert_frame_equal(pd.read_parquet(old/period/cost/f'{control}_ledger.parquet'),
                    pd.read_parquet(folder/f'{control}_ledger.parquet'))
            cache[period, cost] = (plans, factors)
        plans, factors = cache[period, cost]
        targets = np.full(len(frame), np.nan)
        targets[indices] = sum(cfg['source_weights'][candidate][m]*plans[m] for m in MODELS)
        np.testing.assert_allclose(factors[candidate+'_target'], targets, atol=1e-12, rtol=0, equal_nan=True)
        ledger = pd.read_parquet(folder/f'{candidate}_ledger.parquet')
        decisions = pd.read_parquet(folder/f'{candidate}_decisions.parquet')
        require(pd.DatetimeIndex(decisions.decision_time).equals(pd.DatetimeIndex(decisions.origin)+pd.Timedelta(hours=15, minutes=5)), '合并账户决定时钟不同')
        checks.append({'model': candidate, 'period': period, 'cost': cost,
            'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost]),
            'source_zero_but_holding_origins': int(factors.loc[indices, [m+'_zero_signal_but_holding_plan' for m in MODELS]].max(axis=1).eq(1).sum())})
        return targets

    accounts, cycles, differences, count = [], [], [], 0
    for model in CANDIDATES:
        a, c, d, n = verify_saved_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
            lambda p, c, f, s: expected(p, c, f, s, model), comparison_models=list(CONTROLS))
        for destination, rows in [(accounts, a), (cycles, c), (differences, d)]:
            destination.extend({'model': model, **row} for row in rows)
        count += n
    for name, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
        ('saved_comparison_differences.csv', differences), ('saved_source_plan_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')
    require(len(accounts) == 8 and count == 11292, '八账户核对范围不同')
    receipt = {'verified_at': now(), 'status': 'PASS_EIGHT_NETTED_INTENT_ACCOUNTS_AND_PRIOR_CLOSE_SOURCE_PLANS',
        'actual_accounts': 8, 'actual_decisions_checked': count, 'complete_actual_cycles': len(cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'new_models_or_accounts': 0,
        'independent_performance_validation': False, 'security_audit_performed': False,
        'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare': prepare, 'run': run, 'verify': verify}[sys.argv[1]]()
