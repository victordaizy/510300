"""一次批量筛选保存净收益的互补性；输出仅为虚拟诊断，不生成交易账户。"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_saved_combination_fast_screen_20260913'
OLD = ROOT/'reports/research/510300_joint_saved_frontier_through186/candidate_sources.json'
PHASES = [('evaluation', 'BASE'), ('evaluation', 'STRESS'), ('earlier_diagnostic', 'BASE'), ('earlier_diagnostic', 'STRESS')]
NAMES = ['主历史基础', '主历史压力', '较早历史基础', '较早历史压力']


def metrics(values):
    return {'net_sharpe': float(values.mean()/values.std(ddof=1)*np.sqrt(242)),
            'annualized_return': float(np.expm1(np.log1p(values).sum()*242/len(values)))}


def main():
    began = time.perf_counter()
    require(not OUT.exists(), '本诊断已经开始，请接续保存文件')
    OUT.mkdir()
    index_path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 195, '来源截止轮次不同')
    write_json(OUT/'STARTED.json', {'started_at': now(), 'source_through_round': 195,
        'protocol': 'docs/510300_SAVED_COMBINATION_FAST_SCREEN_NEXT_20260913.md',
        'protocol_sha256': digest(ROOT/'docs/510300_SAVED_COMBINATION_FAST_SCREEN_NEXT_20260913.md'),
        'script_sha256': digest(Path(__file__)), 'max_optimizer_starts': 2, 'max_iterations_per_start': 200}, exclusive=True)
    candidates = json.loads(OLD.read_text(encoding='utf-8'))
    for record in index['completed_rounds']:
        if record['round'] <= 186:
            continue
        path = ROOT/record['result']
        result = json.loads(path.read_text(encoding='utf-8'))
        for model in sorted({m['model'] for m in result.get('all_metrics', [])}):
            main_rows = [m for m in result['all_metrics'] if m['model'] == model]
            earlier = [m for m in result.get('earlier_diagnostics', []) if m['model'] == model]
            candidates.append({'model': model, 'name': main_rows[0].get('name', model),
                'assumption': 'NEXT_OPEN' if all(not m.get('assumption') or m['assumption'] == 'NEXT_OPEN' for m in main_rows+earlier) else 'OTHER',
                'main': main_rows, 'earlier': earlier,
                'sources': [{'round': record['round'], 'folder': str(path.parent.relative_to(ROOT)), 'earlier_source': record['result']}]})
    data = pd.read_parquet(ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    calendars = [pd.DatetimeIndex(data.loc[data.date.between(left, right), 'date']) for left, right in
                 [('2020-01-02', '2026-08-14'), ('2015-01-05', '2019-12-31')]]
    included, skipped, duplicates, hashes, matrices, cached = [], [], [], {}, [], {}
    expected_costs = {'BASE': {'commission': .0002, 'minimum': 5., 'slippage': .0005},
                      'STRESS': {'commission': .0004, 'minimum': 5., 'slippage': .001}}
    for number, candidate in enumerate(candidates):
        model = candidate['model']
        if candidate['assumption'] != 'NEXT_OPEN' or len(candidate['main']) != 2 or len(candidate['earlier']) != 2:
            skipped.append({'input': number, 'model': model, 'reason': '非标准完整四场景'})
            continue
        readings, source_used, reason = None, None, '找不到指标相符的完整来源'
        for source in sorted(candidate['sources'], key=lambda s: s['round'], reverse=True):
            if not source.get('earlier_source'):
                continue
            folder = ROOT/source['folder']
            cfg_path = ROOT/'config'/f'{folder.name}.json'
            paths = [folder/period/cost/f'{model}_ledger.parquet' for period, cost in PHASES]
            if not cfg_path.is_file() or not all(p.is_file() for p in paths):
                continue
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            if cfg.get('initial_capital') != 200000 or cfg.get('annual_days') != 242 or cfg.get('cash_annual_rate_assumption') != 0 or cfg.get('costs') != expected_costs:
                reason = '本金、年化或费用不同'
                continue
            readings = []
            for j, ((period, cost), path) in enumerate(zip(PHASES, paths)):
                key = str(path)
                if key not in cached:
                    ledger = pd.read_parquet(path, columns=['date', 'net_return', 'equity'])
                    cached[key] = ledger
                ledger = cached[key]
                stored = next(m for m in candidate['main' if j < 2 else 'earlier'] if m['cost'] == cost)
                r = ledger.net_return.to_numpy(float)
                if not pd.DatetimeIndex(ledger.date).equals(calendars[j//2]) or not np.isfinite(r).all() or np.any(r <= -1) or np.std(r, ddof=1) == 0:
                    readings = None
                    break
                measured = metrics(r)
                if any(stored.get(k) is None or abs(measured[k]-stored[k]) > 1e-7 for k in measured):
                    readings = None
                    break
                np.testing.assert_allclose(ledger.equity/np.r_[200000., ledger.equity.iloc[:-1]]-1, r, atol=1e-12, rtol=0)
                readings.append(r)
            if readings is not None:
                source_used = {**source, 'ledger_files': [str(p.relative_to(ROOT)) for p in paths]}
                break
        if readings is None:
            skipped.append({'input': number, 'model': model, 'reason': reason})
            continue
        identity = hashlib.sha256(np.concatenate(readings).astype('<f8').tobytes()).hexdigest()
        if identity in hashes:
            duplicates.append({'input': number, 'model': model, 'same_as': hashes[identity]})
            continue
        hashes[identity] = len(included)
        included.append({'column': len(included), 'model': model, 'name': candidate['name'], 'source': source_used, 'returns_sha256': identity})
        matrices.append(readings)
    require(len(included) >= 2, '可用不同来源不足')
    returns = [np.column_stack([m[j] for m in matrices]) for j in range(4)]
    np.savez_compressed(OUT/'saved_net_return_matrices.npz', **{f'scenario_{j}': matrix for j, matrix in enumerate(returns)})
    write_json(OUT/'included_sources.json', included, exclusive=True)
    pd.DataFrame(skipped).to_csv(OUT/'excluded_sources.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(duplicates).to_csv(OUT/'duplicate_sources.csv', index=False, encoding='utf-8-sig')
    means = [r.mean(axis=0) for r in returns]
    covariances = [np.cov(r, rowvar=False, ddof=1) for r in returns]
    dimension = len(included)
    print(f'已复用{dimension}条不同的完整策略路径，排除{len(skipped)}项，去除完全重复路径{len(duplicates)}项；开始两次固定起点求解。', flush=True)

    def ratios_and_gradient(weights):
        values, gradients = [], []
        for r, mean, covariance in zip(returns, means, covariances):
            mixed = r @ weights
            sigma = math.sqrt(max(float(weights @ covariance @ weights), 1e-24))
            average = float(mean @ weights)
            annual = float(np.expm1(np.log1p(mixed).sum()*242/len(r)))
            values.extend([average/sigma*np.sqrt(242)/1.2, annual/.1])
            gradients.extend([np.sqrt(242)/1.2*(mean/sigma-average*(covariance @ weights)/sigma**3),
                              (annual+1)*242/len(r)*(r.T @ (1/(1+mixed)))/.1])
        return np.asarray(values), np.asarray(gradients)

    def constraint(x):
        values, _ = ratios_and_gradient(x[:-1])
        return values-x[-1]

    def jacobian(x):
        _, jac = ratios_and_gradient(x[:-1])
        return np.column_stack([jac, -np.ones(8)])

    core = next(i for i, item in enumerate(included) if item['model'] == 'ACCOUNT_VOLATILITY_EXPOSURE')
    starts = [('所有来源等权', np.full(dimension, 1/dimension)), ('原181单独', np.eye(1, dimension, core).ravel())]
    solutions = []
    for label, weights in starts:
        start_time = time.perf_counter()
        initial_values, _ = ratios_and_gradient(weights)
        x0 = np.r_[weights, min(initial_values)-1e-5]
        fit = minimize(lambda x: -x[-1], x0, method='SLSQP', jac=lambda x: np.r_[np.zeros(dimension), -1.],
            bounds=[(0., 1.)]*dimension+[(None, 3.)],
            constraints=[{'type': 'eq', 'fun': lambda x: x[:-1].sum()-1,
                          'jac': lambda x: np.r_[np.ones(dimension), 0.]},
                         {'type': 'ineq', 'fun': constraint, 'jac': jacobian}],
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False})
        w = np.maximum(fit.x[:-1], 0.)
        w /= w.sum()
        values, _ = ratios_and_gradient(w)
        row = {'start': label, 'success': bool(fit.success), 'message': str(fit.message), 'iterations': int(fit.nit),
            'objective_evaluations': int(fit.nfev), 'minimum_joint_ratio': float(min(values)),
            'weights': w.tolist(), 'metrics': [{'scenario': name, **metrics(r @ w)} for name, r in zip(NAMES, returns)],
            'seconds': time.perf_counter()-start_time, 'maximum_constraint_violation': float(max(0., -min(constraint(np.r_[w, fit.x[-1]]))))}
        solutions.append(row)
        write_json(OUT/f'optimizer_{len(solutions)}.json', row, exclusive=True)
        print(f'{label}：最弱门槛比值{min(values):.4f}，迭代{fit.nit}次，收敛状态{fit.success}。', flush=True)
    best = max(solutions, key=lambda row: row['minimum_joint_ratio'])
    weights = np.array(best['weights'])
    order = np.argsort(-weights, kind='stable')
    top = order[:3]
    simplified = np.zeros(dimension)
    simplified[top] = weights[top]/weights[top].sum()
    values, _ = ratios_and_gradient(simplified)
    result = {'completed_at': now(), 'status': 'COMPLETED_RETROSPECTIVE_VIRTUAL_RETURN_SCREEN_NOT_ACCOUNT_BACKTEST',
        'input_records': len(candidates), 'distinct_eligible_paths': dimension, 'duplicate_paths': len(duplicates),
        'excluded_records': len(skipped), 'ledger_files_read': len(cached), 'optimizer_starts': 2,
        'optimizer_objective_evaluations': sum(row['objective_evaluations'] for row in solutions),
        'best_virtual_minimum_joint_ratio': best['minimum_joint_ratio'], 'best_start': best['start'],
        'best_weights': [{'weight': float(weights[j]), **included[j]} for j in order if weights[j] > 1e-6],
        'top_three_renormalized': [{'weight': float(simplified[j]), **included[j]} for j in top],
        'top_three_virtual_minimum_joint_ratio': float(min(values)),
        'top_three_virtual_metrics': [{'scenario': name, **metrics(r @ simplified)} for name, r in zip(NAMES, returns)],
        'new_accounts': 0, 'new_prediction_models': 0, 'goal_achieved': False,
        'independent_performance_validation': False, 'run_seconds': time.perf_counter()-began,
        'limitation': '对重复使用历史的虚拟日净收益做静态权重优化，未模拟组合资金和成交；找到的数值解不是独立验证，未达标也不是不可行性证明。'}
    write_json(OUT/'result.json', result, exclusive=True)
    lines = ['# 保存收益组合的批量快筛', '',
        f"共检查{len(candidates)}条来源记录，保留{dimension}条不同的完整四场景路径，直接读取{len(cached)}份保存账本；没有重跑旧账户。两次预定起点共调用目标函数{result['optimizer_objective_evaluations']}次。耗时{result['run_seconds']:.2f}秒。", '',
        result['limitation'], '',
        f"找到的最佳虚拟组合最弱门槛比值为{best['minimum_joint_ratio']:.4f}；仅保留权重前三个来源并重新归一后为{min(values):.4f}。比值一表示八项指标均达到各自门槛；它不是成功概率。", '',
        '|虚拟组合来源|优化权重|只留前三项后的权重|', '|---|---:|---:|']
    for j in order:
        if weights[j] > 1e-6:
            lines.append(f"|{included[j]['name']}|{weights[j]:.4%}|{simplified[j]:.4%}|")
    lines += ['', '|虚拟组合|历史与成本|夏普|复合年化|', '|---|---|---:|---:|']
    for label, rows in [('完整权重', best['metrics']), ('前三来源', result['top_three_virtual_metrics'])]:
        lines += [f"|{label}|{r['scenario']}|{r['net_sharpe']:.4f}|{r['annualized_return']:.2%}|" for r in rows]
    lines += ['', '目标尚未完成。可取的来源和权重需要另行落实为明确进出场、整手与现金约束的完整账户；本表不能用于替代它们。']
    (OUT/'批量组合快筛说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    index['latest_saved_combination_screen'] = str((OUT/'result.json').relative_to(ROOT))
    index['next_work']['preflight_completed'] = True
    write_json(index_path, index)
    print(json.dumps({k: v for k, v in result.items() if k not in ['best_weights', 'top_three_renormalized']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
