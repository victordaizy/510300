"""只用已保存账户拆分算术平均收益差，不生成新策略、账户或拟合。"""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_monthly_downside_forecast_v1'


def main():
    result = json.loads((OUT / 'result.json').read_text(encoding='utf-8-sig'))
    config = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8-sig'))
    destination = OUT / 'saved_arithmetic_return_attribution.json'
    assert not destination.exists(), '已完成保存收益归因，不重复计算'
    data = pd.read_parquet(ROOT / config['features']).set_index('date')
    components = {}
    for item in result['metrics']:
        source = ROOT / item['source'] if item['reused'] else OUT / 'accounts' / item['period'] / item['cost'] / item['model'] / 'ledger.parquet'
        ledger = pd.read_parquet(source)
        market = data.loc[ledger.date]
        previous_nav = np.r_[config['initial_capital'], ledger.equity.to_numpy()[:-1]]
        prior_weight = ledger.shares_before.to_numpy() * market.previous_close.to_numpy() / previous_nav
        market_return = market.total_simple.to_numpy()
        actual = ledger.net_return.to_numpy()
        exposure = float(prior_weight.mean() * market_return.mean() * 242)
        timing = float(np.mean((prior_weight - prior_weight.mean()) * market_return) * 242)
        execution_residual = float(np.mean(actual - prior_weight * market_return) * 242)
        annual_mean = float(actual.mean() * 242)
        np.testing.assert_allclose(exposure + timing + execution_residual, annual_mean, atol=1e-12, rtol=0)
        components[item['period'], item['cost'], item['model']] = {
            'mean_prior_weight': float(prior_weight.mean()), 'exposure_component': exposure,
            'timing_component': timing, 'execution_and_rights_residual': execution_residual,
            'annual_arithmetic_mean': annual_mean,
        }
    rows = []
    for period in config['periods']:
        for cost in config['costs']:
            candidate = components[period, cost, 'LOG_RISK_RIDGE']
            for comparator in ['PAST_MONTHLY_MEAN', 'RECENT_DOWNSIDE20', 'BUY_HOLD', 'MONTHLY_VOL10']:
                baseline = components[period, cost, comparator]
                rows.append({'period': period, 'cost': cost, 'comparator': comparator,
                             'candidate': candidate, 'baseline': baseline,
                             'difference': {key: candidate[key] - baseline[key] for key in candidate}})
    saved = {
        'completed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'COMPLETED_SAVED_ACCOUNT_ARITHMETIC_IDENTITY',
        'scope': '解释既有年化算术平均日收益差；不是复合年化收益拆分，不证明因果，不生成常仓位反事实账户',
        'components': '平均前日仓位乘市场平均收益；仓位偏离平均值与市场日收益的协同项；调仓时点、成本及分红权利等剩余项',
        'comparisons': rows, 'new_accounts': 0, 'new_model_fits': 0,
        'new_candidate_methods': 0, 'goal_achieved': False,
    }
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(saved, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps([r for r in rows if r['period'] == 'main' and r['cost'] == 'STRESS'
                      and r['comparator'] == 'MONTHLY_VOL10'], ensure_ascii=True))


if __name__ == '__main__':
    main()
