"""复用258条保存矩阵，只增量读最近50个新候选的四场景账本。"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.saved_return_mixture_optimizer_v1 import fit_static_mixtures
from scripts.saved_combination_fast_screen_20260913 import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_incremental_saved_mix_through208'
OLD = ROOT/'reports/research/510300_incremental_saved_mix_through199'
PHASES = [('evaluation', 'BASE'), ('evaluation', 'STRESS'), ('earlier_diagnostic', 'BASE'), ('earlier_diagnostic', 'STRESS')]
TAGS = ['主历史基础', '主历史压力', '较早历史基础', '较早历史压力']


def main():
    began = time.perf_counter()
    require(not OUT.exists(), '增量组合诊断已经开始')
    OUT.mkdir()
    source_folders = ['510300_three_source_order_intent_mix_v1', '510300_finite_rebalance_band_batch_v1',
        '510300_account_cycle_loss_exit_batch_v1', '510300_close_return_buy_gate_batch_v1',
        '510300_close_return_buy_strength_gate_batch_v1', '510300_addition_only_return_gate_batch_v1',
        '510300_addition_gate_exposure_batch_v1', '510300_positive_target_reduction_batch_v1',
        '510300_gated_source_ablation_batch_v1']
    write_json(OUT/'STARTED.json', {'started_at': now(), 'reuse': str(OLD.relative_to(ROOT)),
        'incremental_source_folders': source_folders, 'starts': ['全部可用来源等权', '原205单独'],
        'max_iterations_per_start': 200, 'source_read_limit_new_candidates': 50,
        'objective': '四场景夏普与年化门槛比值的最小值', 'weights': '非负且总和为一',
        'new_trading_accounts': 0, 'new_prediction_models': 0,
        'classification': 'RETROSPECTIVE_VIRTUAL_RETURN_COMBINATION_NOT_EXECUTABLE_ACCOUNT',
        'script_sha256': digest(Path(__file__)), 'optimizer_sha256': digest(ROOT/'research/saved_return_mixture_optimizer_v1.py')}, exclusive=True)
    included = json.loads((OLD/'included_sources.json').read_text(encoding='utf-8'))
    with np.load(OLD/'saved_return_matrices.npz') as saved:
        matrices = [saved[f'scenario_{j}'].copy() for j in range(4)]
    require(len(included) == 258 and all(m.shape[1] == 258 for m in matrices), '旧保存矩阵范围不同')
    identities = {hashlib.sha256(np.concatenate([m[:, j] for m in matrices]).astype('<f8').tobytes()).hexdigest(): j for j in range(258)}
    require(all(item['returns_sha256'] in identities for item in included), '旧矩阵与来源身份不同')
    data = pd.read_parquet(ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    dates = [pd.DatetimeIndex(data.loc[data.date.between(left, right), 'date']) for left, right in
             [('2020-01-02', '2026-08-14'), ('2015-01-05', '2019-12-31')]]
    source_files = [OLD/'included_sources.json', OLD/'saved_return_matrices.npz']
    added, duplicates, inspected = 0, [], 0
    for name in source_folders:
        folder = ROOT/'reports/research'/name
        cfg = json.loads((ROOT/'config'/f'{name}.json').read_text(encoding='utf-8'))
        result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
        require((folder/'saved_verification_receipt.json').exists(), '增量账户尚未完成保存核对')
        source_files += [folder/'result.json', folder/'saved_verification_receipt.json', ROOT/'config'/f'{name}.json']
        require(cfg['initial_capital'] == 200000 and cfg['annual_days'] == 242 and cfg['cash_annual_rate_assumption'] == 0, '增量账户口径不同')
        for model in cfg['candidate_models']:
            streams, paths = [], []
            for j, (period, cost) in enumerate(PHASES):
                path = folder/period/cost/f'{model}_ledger.parquet'
                ledger = pd.read_parquet(path, columns=['date', 'net_return', 'equity'])
                require(pd.DatetimeIndex(ledger.date).equals(dates[j//2]), '新增路径完整日期不同')
                r = ledger.net_return.to_numpy(float)
                np.testing.assert_allclose(ledger.equity/np.r_[200000., ledger.equity.iloc[:-1]]-1, r, atol=1e-12, rtol=0)
                stored = next(m for m in result['all_metrics' if j < 2 else 'earlier_diagnostics'] if m['model'] == model and m['cost'] == cost)
                for key, value in metrics(r).items():
                    require(abs(value-stored[key]) < 1e-7, '新增保存收益不能复算')
                streams.append(r)
                paths.append(path)
            inspected += 1
            identity = hashlib.sha256(np.concatenate(streams).astype('<f8').tobytes()).hexdigest()
            if identity in identities:
                duplicates.append({'model': model, 'same_as_column': identities[identity]})
                continue
            column = len(included)
            identities[identity] = column
            included.append({'column': column, 'model': model, 'name': stored['name'], 'returns_sha256': identity,
                'source': {'round': cfg['round'], 'folder': str(folder.relative_to(ROOT)),
                    'ledger_files': [str(p.relative_to(ROOT)) for p in paths]}})
            matrices = [np.column_stack([matrix, stream]) for matrix, stream in zip(matrices, streams)]
            source_files += paths
            added += 1
    require(inspected == 50, '本次增量候选数量不同')
    write_json(OUT/'included_sources.json', included, exclusive=True)
    np.savez_compressed(OUT/'saved_return_matrices.npz', **{f'scenario_{j}': m for j, m in enumerate(matrices)})
    n = len(included)
    baseline = next(item['column'] for item in included if item['model'] == 'ADD_GATE_BAND20_LOWER01')
    starts = [('全部可用来源等权', np.full(n, 1/n)), ('原205单独', np.eye(1, n, baseline).ravel())]
    print(f'复用258条矩阵，增量核对{inspected*4}份账本，新增{added}条不同路径；开始两个固定起点。', flush=True)
    solutions = fit_static_mixtures(matrices, starts)
    for i, solution in enumerate(solutions, 1):
        write_json(OUT/f'optimizer_{i}.json', solution, exclusive=True)
    best = max(solutions, key=lambda s: s['minimum_joint_ratio'])
    weights = np.array(best['weights'])
    order = np.argsort(-weights, kind='stable')
    simplified = np.zeros(n)
    top = order[:3]
    simplified[top] = weights[top]/weights[top].sum()
    records = []
    for label, w in [('完整虚拟权重', weights), ('前三来源归一化', simplified)]:
        for tag, matrix in zip(TAGS, matrices):
            records.append({'variant': label, 'scenario': tag, **metrics(matrix @ w)})
    simplified_ratio = min(min(r['net_sharpe']/1.2, r['annualized_return']/.1) for r in records if r['variant'] == '前三来源归一化')
    pd.DataFrame(records).to_csv(OUT/'virtual_metrics.csv', index=False, encoding='utf-8-sig')
    write_json(OUT/'source_files.json', [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(source_files))], exclusive=True)
    summary = {'completed_at': now(), 'status': 'COMPLETED_INCREMENTAL_VIRTUAL_MIX_NOT_ACCOUNT_BACKTEST',
        'reused_paths': 258, 'new_candidates_read': inspected, 'new_ledger_files_read': inspected*4,
        'new_distinct_paths': added, 'total_paths': n, 'duplicates': duplicates, 'optimizer_starts': 2,
        'optimizer_objective_evaluations': sum(s['objective_evaluations'] for s in solutions),
        'best_virtual_minimum_joint_ratio': best['minimum_joint_ratio'], 'top_three_virtual_minimum_joint_ratio': simplified_ratio,
        'best_weights': [{'weight': float(weights[j]), **included[j]} for j in order if weights[j] > 1e-6],
        'metrics': records, 'new_accounts': 0, 'new_prediction_models': 0, 'goal_achieved': False,
        'independent_validation': False, 'run_seconds': time.perf_counter()-began,
        'limitation': '使用过历史后的理想日净收益混合，未模拟组合资金和成交，不能把这些指标当作可交易策略；数值未找到达标也不构成不可行性证明。'}
    write_json(OUT/'result.json', summary, exclusive=True)
    lines = ['# 截至208轮的增量虚拟组合筛选', '',
        f"复用258条既有矩阵，仅新增读取{inspected*4}份保存账本；当前{n}条不同完整四场景路径。两次固定起点共{summary['optimizer_objective_evaluations']}次目标函数评价，总耗时{summary['run_seconds']:.2f}秒。没有生成新的交易账户。", '',
        summary['limitation'], '',
        f"找到的最佳虚拟最弱门槛比值为{best['minimum_joint_ratio']:.4f}，仅保留前三来源后为{simplified_ratio:.4f}。一代表八项指标均达到各自门槛，该数不是胜率。", '',
        '|来源|虚拟权重|', '|---|---:|']
    lines += [f"|{included[j]['name']}|{weights[j]:.4%}|" for j in order if weights[j] > 1e-6]
    lines += ['', '|虚拟方案|历史与成本|夏普|复合年化|', '|---|---|---:|---:|']
    lines += [f"|{r['variant']}|{r['scenario']}|{r['net_sharpe']:.4f}|{r['annualized_return']:.2%}|" for r in records]
    (OUT/'增量组合筛选说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['best_weights', 'duplicates']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
