"""按实际入场后的交易日龄汇总已存账本，定位下一步而不重跑账户。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/research/510300_finite_rebalance_band_batch_v1'
OUT = ROOT/'reports/research/510300_saved_entry_age_attribution_through201'
MODEL = 'INTENT_MIX_BAND_00'


def main():
    require(not OUT.exists(), '入场日龄诊断已经保存，不重复运行')
    require((SOURCE/'saved_verification_receipt.json').exists(), '来源必要核对未完成')
    OUT.mkdir(parents=True)
    rows, cycles, inputs = [], [], []
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in ['BASE', 'STRESS']:
            path = SOURCE/period/cost/f'{MODEL}_ledger.parquet'
            ledger = pd.read_parquet(path)
            inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
            old_shares, entry, cycle = 0, None, 0
            day_records, cycle_days = [], []
            for i, row in enumerate(ledger.itertuples()):
                if old_shares == 0 and row.shares > 0:
                    entry, cycle = i, cycle+1
                    cycle_days = []
                age = i-entry+1 if entry is not None else 0
                phase = '入场首日' if age == 1 else '第2至5日' if 2 <= age <= 5 else '第6至10日' if 6 <= age <= 10 else '第11日以后' if age > 10 else '空仓及应收结算'
                value = {'period': period, 'cost': cost, 'date': row.date, 'cycle': cycle if age else 0,
                    'age': age, 'phase': phase, 'net_pnl': row.pnl, 'price_pnl': row.price_pnl,
                    'dividend': row.dividend_recognized, 'costs': row.commission+row.slippage_cost,
                    'net_return': row.net_return, 'negative_pnl': min(0., row.pnl),
                    'positive_pnl': max(0., row.pnl), 'held_at_close': row.shares > 0}
                day_records.append(value)
                if age:
                    cycle_days.append(value)
                    if row.shares == 0:
                        cycles.append({'period': period, 'cost': cost, 'cycle': cycle,
                            'entry_date': cycle_days[0]['date'], 'exit_date': row.date, 'calendar_trading_rows': age,
                            'first_day_pnl': cycle_days[0]['net_pnl'],
                            'first_five_days_pnl': sum(r['net_pnl'] for r in cycle_days if r['age'] <= 5),
                            'later_days_pnl': sum(r['net_pnl'] for r in cycle_days if r['age'] > 5),
                            'net_profit': sum(r['net_pnl'] for r in cycle_days)})
                        entry = None
                old_shares = row.shares
            detail = pd.DataFrame(day_records)
            require(np.isclose(detail.net_pnl.sum(), ledger.equity.iloc[-1]-200000., atol=1e-6, rtol=0), '日龄汇总不能还原全账户')
            for phase, group in detail.groupby('phase', sort=False):
                rows.append({'period': period, 'cost': cost, 'phase': phase, 'rows': len(group),
                    'negative_days': int(group.net_pnl.lt(0).sum()), 'positive_days': int(group.net_pnl.gt(0).sum()),
                    **{k: float(group[k].sum()) for k in ['net_pnl', 'price_pnl', 'dividend', 'costs', 'negative_pnl', 'positive_pnl']}})
    pd.DataFrame(rows).to_csv(OUT/'phase_attribution.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cycles).to_csv(OUT/'cycle_entry_age.csv', index=False, encoding='utf-8-sig')
    result = {'completed_at': now(), 'status': 'COMPLETED_SAVED_ENTRY_AGE_ATTRIBUTION_NOT_NEW_BACKTEST',
        'model': MODEL, 'ledger_files_read': 4, 'complete_cycles': len(cycles), 'new_accounts': 0,
        'goal_achieved': False, 'inputs': inputs, 'phase_attribution': rows,
        'limitation': '入场日龄是保存资金曲线的事后拆解，不是延后进入或止损的收益。不同阶段资金规模和机会不同，不能删掉不利日、只选择最终亏损交易、或假设改动后其余资金路径不变。'}
    write_json(OUT/'result.json', result, exclusive=True)
    lines = ['# 第201轮保存账户入场日龄归因', '', result['limitation'], '',
        '|区间|费用|入场后阶段|日数|净利润|其中负利润合计|交易费用|', '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['phase']}|{row['rows']}|{row['net_pnl']:.2f}|{row['negative_pnl']:.2f}|{row['costs']:.2f}|")
    (OUT/'入场日龄归因.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'阶段汇总': rows, '完整周期': len(cycles), '新增账户': 0}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
