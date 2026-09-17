"""初始资金和主起点的完整依赖图敏感性，使用独立输出。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from research.strategy_review_diagnostics_v1 import ROOT, OUT, MAIN, INPUTS, MODEL, read, write, metrics, inputs
from research.post_selection_continuous_replay_v1 import Pipeline
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account, simulate_policy, simulate_rearmed_exit
from research.post_selection_continuous_factors_v1 import align_decisions
from research.simple_intraday_protection_v1 import make_rules
from research.adaptive_allocation_v1 import target_request


class DiagnosticPipeline(Pipeline):
    def __init__(self, variant):
        self.variant = variant
        self.destination = OUT / 'full_graph' / variant
        self.destination.mkdir(parents=True, exist_ok=True)
        self.cfg, self.data, self.div = inputs()
        self.capital = {'CAPITAL_100K': 100000., 'CAPITAL_1M': 1000000., 'RESET_2021': 200000., 'BASELINE_RECONSTRUCTION': 200000.}[variant]
        self.start = '2021-01-04' if variant == 'RESET_2021' else '2020-01-02'
        self.cfg['initial_capital'] = self.capital
        self.first = int(np.flatnonzero(self.data.date.ge(self.start))[0])
        self.next_date = '2026-09-14'
        self.graph = {n['node']: deepcopy(n) for n in read(INPUTS / 'dependency_graph.json')['nodes']}
        if variant == 'RESET_2021':
            for node in self.graph.values():
                if node['replay_start'] == '2020-01-02':
                    node['replay_start'] = self.start
        self.accounts, self.targets, self.settings, self.calls = {}, {}, {}, {}
        self.checks, self.target_checks = [], []
        self.within = read(MAIN / 'within_models.json')['models']
        self.ridge = read(MAIN / 'ridge_models.json')['models']
        self.session = make_rules(self.data)['D60_INTRA']

    def setting(self, key):
        if key not in self.settings:
            cfg = read(ROOT / self.graph[key]['configuration'])
            cfg['initial_capital'] = self.capital
            if self.variant == 'RESET_2021' and cfg.get('evaluation_start') == '2020-01-02':
                cfg['evaluation_start'] = self.start
            self.settings[key] = cfg
        return self.settings[key]

    def target(self, key, cost, values, check=True):
        values = np.asarray(values, float)
        assert len(values) == len(self.data)
        assert (np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), key
        self.targets[key, cost] = values
        return values

    def save_factors(self, name, frame):
        folder = self.destination / 'factors'
        folder.mkdir(exist_ok=True)
        frame.to_parquet(folder / (name+'.parquet'), index=False)

    def account(self, key, cost, kind='target', values=None, rule=None, spec=None, controller_factory=None, request=None):
        assert (key, cost) not in self.accounts
        folder = self.destination / 'accounts' / cost / key
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'checkpoint.json').exists():
            ledger = pd.read_parquet(folder / 'ledger.parquet')
            decisions = pd.read_parquet(folder / 'decisions.parquet')
            state = read(folder / 'checkpoint.json')
        else:
            cfg = self.setting(key)
            start = self.graph[key]['replay_start']
            args = (self.data, self.div, cfg, cfg['costs'][cost], start)
            kwargs = {'next_execution_date': self.next_date}
            if kind == 'target':
                fn = simulate_indexed_request_account
                args += (key,)
                kwargs.update(targets=values, event_mask=np.ones(len(self.data), bool),
                              request_policy=request or (lambda a, p, v, c, m, t: target_request(a, p, v, c)))
            else:
                fn = simulate_rearmed_exit if kind == 'rearmed' else simulate_policy
                args += (rule, spec)
                if kind == 'rearmed':
                    kwargs['controller'] = controller_factory()
            result = fn(*args, **kwargs)
            ledger, decisions, state = result[0], result[1], result[-1]
            decisions['decision_time'] = pd.to_datetime(decisions.origin)+pd.Timedelta(hours=15, minutes=5)
            decisions['source_model'], decisions['source_cost'] = key, cost
            assert ledger.accounting_error.abs().max() < 1e-6
            ledger.to_parquet(folder / 'ledger.parquet', index=False)
            decisions.to_parquet(folder / 'decisions.parquet', index=False)
            if kind != 'target':
                result[2].to_csv(folder / 'cycles.csv', index=False, encoding='utf-8-sig')
            write(folder / 'checkpoint.json', state)
        self.accounts[key, cost] = ledger, decisions, state
        self.target(key, cost, align_decisions(self.data, decisions), check=False)
        print(f'完整依赖图诊断 {self.variant}：{len(self.accounts)}/22，{key}／{cost}。', flush=True)
        return ledger, decisions, state


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='固定模型的完整依赖图资金／冷启动敏感性')
    p.add_argument('variant', choices=['CAPITAL_100K', 'CAPITAL_1M', 'RESET_2021', 'BASELINE_RECONSTRUCTION'])
    variant = p.parse_args().variant
    if not (OUT / 'protocol_freeze.json').exists():
        raise RuntimeError('需要先冻结本轮诊断协议。')
    destination = OUT / 'full_graph' / variant
    if (destination / 'result.json').exists():
        print('该完整依赖图诊断已完成，直接复用结果。', flush=True)
        return
    started = time.perf_counter()
    pipeline = DiagnosticPipeline(variant).run()
    if variant == 'BASELINE_RECONSTRUCTION':
        checks = []
        for (node, cost), (ledger, decisions, state) in pipeline.accounts.items():
            reference = pd.read_parquet(MAIN / 'accounts' / cost / node / 'ledger.parquet')
            assert len(reference) == len(ledger)
            for column in ['cash', 'shares', 'equity', 'net_return', 'requested_quantity', 'filled_quantity']:
                np.testing.assert_allclose(ledger[column], reference[column], rtol=0, atol=1e-7)
            checks.append({'node': node, 'cost': cost, 'rows': len(ledger),
                           'maximum_equity_difference': float(np.max(np.abs(ledger.equity-reference.equity)))})
        write(OUT / 'full_graph_reconstruction_checks.json', {'status': 'PASS_ALL_22_ORIGINAL_ACCOUNT_RECONSTRUCTION', 'checks': checks})
    rows, differences = [], []
    for cost in pipeline.cfg['costs']:
        ledger, decisions, state = pipeline.accounts[MODEL, cost]
        rows.append({'variant': variant, 'cost': cost, 'capital': pipeline.capital, **metrics(ledger, pipeline.capital)})
        original = pd.read_parquet(MAIN / 'accounts' / cost / MODEL / 'decisions.parquet')
        matched = decisions.merge(original, on='origin', suffixes=('_new', '_original'))
        a, b = matched.reference_weight_new.to_numpy(float), matched.reference_weight_original.to_numpy(float)
        valid = np.isfinite(a) & np.isfinite(b)
        differences.append({'variant': variant, 'cost': cost, 'compared_decisions': int(valid.sum()),
                            'target_changed_days': int((np.abs(a[valid]-b[valid]) > 1e-10).sum()),
                            'max_target_difference': float(np.max(np.abs(a[valid]-b[valid]))),
                            'mean_target_difference': float(np.mean(np.abs(a[valid]-b[valid]))),
                            'zero_positive_disagreement_days': int(((a[valid] == 0) != (b[valid] == 0)).sum())})
    write(destination / 'result.json', {'status': 'COMPLETED_FIXED_MODEL_FULL_GRAPH_DIAGNOSTIC', 'variant': variant,
          'accounts': len(pipeline.accounts), 'dependency_nodes': len(pipeline.graph), 'metrics': rows, 'differences': differences,
          'new_model_fits': 0, 'model_training_and_selection_history_fixed': True, 'strict_forward_days': 0,
          'seconds': time.perf_counter()-started, 'completed_at': datetime.now().astimezone().isoformat()})
    print(f'完整依赖图 {variant} 诊断完成。', flush=True)


if __name__ == '__main__':
    main()
