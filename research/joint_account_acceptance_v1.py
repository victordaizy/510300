"""独立呈现夏普与复合年化的联合门槛，不将旧夏普字段当作目标完成。"""
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import now, require, write_json


def save_joint_assessment(out, cfg, result):
    models, costs = cfg['candidate_models'], list(cfg['costs'])
    rows, assessment = [], {}
    for model in models:
        by_period = {}
        for period, key in [('evaluation', 'all_metrics'), ('earlier_diagnostic', 'earlier_diagnostics')]:
            measured = [r for r in result[key] if r['model'] == model]
            require(len(measured)==len(costs) and {r['cost'] for r in measured}==set(costs), '联合验收缺少账户或存在重复')
            passes = []
            for row in measured:
                s, a = row['net_sharpe'], row['annualized_return']
                passed = s is not None and np.isfinite(s) and np.isfinite(a) and s >= cfg['high_sharpe_target'] and a >= cfg['annual_return_target']
                rows.append({'period': period, **row, 'joint_point_pass': bool(passed)})
                passes.append(bool(passed))
            by_period[period] = all(passes)
        assessment[model] = {'main_two_cost_joint_pass': by_period['evaluation'],
                             'earlier_two_cost_joint_pass': by_period['earlier_diagnostic'],
                             'four_scenario_joint_pass': all(by_period.values())}
    pd.DataFrame(rows).to_csv(out/'joint_target_metrics.csv', index=False, encoding='utf-8-sig')
    receipt = {'recorded_at': now(), 'net_sharpe_minimum': cfg['high_sharpe_target'],
               'compound_annual_return_minimum': cfg['annual_return_target'], 'required_costs': costs,
               'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED', 'candidates': assessment}
    write_json(out/'joint_target_assessment.json', receipt, exclusive=True)
    return receipt
