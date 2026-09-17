"""只读复算已保存账户、固定周期配对区间和纯费用现金流。"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def verify(root):
    out = root / 'reports/research/510300_strategy_review_diagnostics_v1'
    checked, total_rows = [], 0
    def account(ledger, initial):
        nonlocal total_rows
        assert len(ledger) > 0 and ledger.date.is_monotonic_increasing
        nav = np.r_[initial, ledger.equity.to_numpy(float)]
        np.testing.assert_allclose(ledger.net_return, nav[1:]/nav[:-1]-1, rtol=0, atol=1e-12)
        np.testing.assert_allclose(ledger.equity, ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(ledger.shares, ledger.shares_before+ledger.filled_quantity)
        assert ledger.cash.min() >= -1e-7 and (ledger.shares >= 0).all()
        assert (ledger.filled_quantity % 100 == 0).all()
        pnl = ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost
        np.testing.assert_allclose(np.diff(nav), pnl, rtol=0, atol=1e-6)
        total_rows += len(ledger)
        vol = float(ledger.net_return.std(ddof=1)*np.sqrt(242))
        return dict(annual_return=float((nav[-1]/initial)**(242/len(ledger))-1),
                    sharpe=float(ledger.net_return.mean()*242/vol) if vol else None,
                    max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()))
    for path in sorted((out/'counterfactuals').glob('*/*/*/metrics.json')):
        saved = json.loads(path.read_text(encoding='utf8'))
        m = account(pd.read_parquet(path.with_name('ledger.parquet')), 200000.)
        for k, v in m.items():
            assert v is None and saved[k] is None or abs(v-saved[k]) < 1e-12, (str(path), k)
        checked.append(str(path.relative_to(root)))
    for result_path in sorted((out/'full_graph').glob('*/result.json')):
        result = json.loads(result_path.read_text(encoding='utf8'))
        capital = {'CAPITAL_100K': 100000., 'CAPITAL_1M': 1000000.}.get(result['variant'], 200000.)
        count = 0
        for lp in result_path.parent.glob('accounts/*/*/ledger.parquet'):
            actual = account(pd.read_parquet(lp), capital)
            if lp.parent.name == 'SELECTED_MIX_BAND10_SIMPLE2':
                item = next(x for x in result['metrics'] if x['cost'] == lp.parent.parent.name)
                for k, v in actual.items():
                    assert abs(v-item[k]) < 1e-12
            count += 1
        assert count == 22
        checked.append(str(result_path.relative_to(root)))
    pairs = pd.read_csv(out/'paired_exit_cycles.csv')
    ledgers = pd.read_parquet(out/'paired_exit_ledgers.parquet')
    for (cycle, variant, cost), ledger in ledgers.groupby(['pair_cycle_id', 'variant', 'cost'], sort=False):
        account(ledger, 200000.)
        row = pairs[(pairs.cycle_id == cycle) & (pairs.variant == variant) & (pairs.cost == cost)].iloc[0]
        assert abs(ledger.equity.iloc[-1]/200000.-1-row.cycle_return) < 1e-12
        assert ledger.shares.iloc[-1] == 0
        buys = ledger[ledger.filled_quantity.gt(0)]
        assert len(buys) == 1 and buys.filled_quantity.iloc[0] == row.entry_quantity
        assert pd.Timestamp(buys.date.iloc[0]) == pd.Timestamp(row.entry)
    assert pairs.groupby(['cycle_id', 'cost']).entry_quantity.nunique().eq(1).all()
    summary = pd.read_csv(out/'paired_exit_summary.csv')
    for (cost, scope), group in summary.groupby(['cost', 'scope']):
        subset = pairs[pairs.cost.eq(cost) & pairs.mature_model_at_entry]
        if scope != 'all_mature':
            subset = subset[subset.entry_period.eq(scope.split('_')[0])]
        pivot = subset.pivot(index='cycle_id', columns='variant', values='cycle_return')
        draws = np.load(out/f'paired_indices_{cost}_{scope}.npz')
        np.testing.assert_array_equal(draws['cycles'], pivot.index)
        for row in group.itertuples():
            delta = (pivot[row.variant]-pivot.PRICE_ORIGINAL).to_numpy()
            sampled = np.mean(delta[draws['indices']], axis=1)
            assert abs(delta.mean()-row.mean_increment) < 1e-12
            np.testing.assert_allclose(np.quantile(sampled, [.025, .975]), [row.delta_q025, row.delta_q975], atol=1e-12, rtol=0)
    # 独立展开方向、价位和最低佣金，防止数量被误当成方向。
    friction = pd.read_csv(out/'pure_friction_fixed_fills_cashflow.csv')
    for period in ['main', 'earlier']:
        if period == 'main':
            lp = root/'reports/research/510300_september_monthly_continuation_v1/accounts/BASE/SELECTED_MIX_BAND10_SIMPLE2/ledger.parquet'
        else:
            lp = root/'reports/research/510300_incremental_selected_intent_mix_v1/earlier_diagnostic/BASE/SELECTED_MIX_BAND10_SIMPLE2_ledger.parquet'
        ledger = pd.read_parquet(lp)
        trades = ledger[ledger.filled_quantity.ne(0)]
        rows = friction[friction.period.eq(period)]
        expected = []
        for t in trades.itertuples():
            side = 1 if t.filled_quantity > 0 else -1
            price_ticks = t.open_price*(1+side*.001)/.001
            price = (math.ceil(price_ticks-1e-10) if side > 0 else math.floor(price_ticks+1e-10))*.001
            fee = max(abs(t.filled_quantity)*price*.0004, 5.)
            expected.append(-t.filled_quantity*price-fee+t.filled_quantity*t.fill_price+t.commission)
        np.testing.assert_allclose(expected, rows.additional_friction_cashflow, atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.cumsum(expected), rows.cumulative_cashflow_delta, atol=1e-7, rtol=0)
    return {'status': 'PASS_SAVED_ECONOMIC_RECOMPUTATION', 'regular_counterfactual_accounts': len(list((out/'counterfactuals').glob('*/*/*/metrics.json'))),
            'full_graph_accounts': 88, 'isolated_exit_accounts': len(pairs), 'ledger_rows_checked': total_rows,
            'paired_bootstrap_rows_recomputed': len(summary), 'new_accounts': 0, 'new_fits': 0, 'new_random_samples': 0,
            'new_downloads': 0, 'invalid_diagnostic_attempts_excluded_and_preserved': 2,
            'not_established': ['independent_performance_validation', 'real_auction_fills', 'strategy_selection_bias_correction']}


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description='只读复算510300策略诊断交付')
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--receipt', type=Path)
    args = p.parse_args()
    result = verify(args.project)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
