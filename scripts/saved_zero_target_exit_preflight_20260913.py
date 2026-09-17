"""复用原181保存退出，描述短暂归零和退出后价格；不是新策略收益。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/research/510300_account_volatility_exposure_v1'
OUT = ROOT/'reports/research/510300_saved_zero_target_exit_preflight_20260913'
MODEL = 'ACCOUNT_VOLATILITY_EXPOSURE'


def main():
    began = time.perf_counter()
    require(not OUT.exists(), '本退出诊断已经保存')
    OUT.mkdir()
    data = pd.read_parquet(ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    events, summaries, sources = [], [], []
    for period in ['evaluation', 'earlier_diagnostic']:
        frame = data if period == 'evaluation' else data[data.date.le('2019-12-31')]
        by_date = {d: j for j, d in enumerate(frame.date)}
        for cost in ['BASE', 'STRESS']:
            lp = SOURCE/period/cost/f'{MODEL}_ledger.parquet'
            dp = SOURCE/period/cost/f'{MODEL}_decisions.parquet'
            sources += [lp, dp]
            ledger, decisions = pd.read_parquet(lp), pd.read_parquet(dp)
            require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(decisions.execution_date)), '退出来源日历错位')
            target = decisions.reference_weight.to_numpy(float)
            zero_runs = []
            k = 1
            while k < len(target):
                if target[k] == 0 and target[k-1] > 0:
                    end = k
                    while end < len(target) and target[end] == 0:
                        end += 1
                    if end < len(target) and target[end] > 0:
                        zero_runs.append(end-k)
                    k = end
                else:
                    k += 1
            current = []
            for k, row in enumerate(ledger.itertuples()):
                if row.filled_quantity >= 0 or row.shares != 0 or target[k] != 0 or row.mark_clock == 'OPEN_TERMINAL':
                    continue
                day = by_date[row.date]
                event = {'period': period, 'cost': cost, 'exit_date': row.date, 'origin': decisions.origin.iloc[k],
                    'sold_quantity': int(-row.filled_quantity), 'exit_open': float(row.open)}
                for delay in [1, 2]:
                    available = day+delay < len(frame)
                    event[f'price_only_change_{delay}day'] = float((-row.filled_quantity)*(frame.open.iloc[day+delay]-row.open)) if available else None
                    event[f'ex_dividend_in_{delay}day_interval'] = bool(frame.dividend.iloc[day+1:day+delay+1].ne(0).any()) if available else None
                current.append(event)
            events.extend(current)
            summary = {'period': period, 'cost': cost, 'observed_explicit_zero_exits': len(current),
                'zero_gaps_followed_by_positive': len(zero_runs), 'one_origin_zero_gaps': zero_runs.count(1),
                'two_origin_zero_gaps': zero_runs.count(2)}
            for delay in [1, 2]:
                values = [e[f'price_only_change_{delay}day'] for e in current if e[f'price_only_change_{delay}day'] is not None]
                summary[f'price_only_change_{delay}day_sum'] = float(sum(values))
                summary[f'price_only_change_{delay}day_positive_count'] = sum(v > 0 for v in values)
                summary[f'available_{delay}day_exits'] = len(values)
            summaries.append(summary)
    pd.DataFrame(events).to_csv(OUT/'saved_exit_price_events.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(OUT/'saved_zero_gap_summary.csv', index=False, encoding='utf-8-sig')
    result = {'completed_at': now(), 'status': 'SAVED_ZERO_GAP_AND_EXIT_PRICE_DESCRIPTION_ONLY',
        'new_accounts': 0, 'new_models': 0, 'goal_achieved': False, 'independent_validation': False,
        'summary': summaries, 'run_seconds': time.perf_counter()-began,
        'limitation': '按原实际卖出份额查看后续开盘价格差，仅作退出研究线索；未纳入分红差、增量资金、再入场、成本和回撤，不是延迟退出策略绩效。',
        'sources': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(sources))]}
    write_json(OUT/'result.json', result, exclusive=True)
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
