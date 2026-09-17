"""核对两条新增反弹参考及八条组合账户，直接读取保存结果，不重跑回测。"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.mean_rebound_auxiliary_v1 import AUX, AUX_SOURCE, CONFIG, CONTROLS, CORE, OUT, PARENTS, REFERENCES, ROOT
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from scripts.finalize_round96_20260908 import value_equal
from scripts.review_round74_saved import saved_cycles


def prices_and_costs(ledger, frame, first, cfg, cost):
    require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(), '现金或份额越界')
    np.testing.assert_allclose(ledger.open, frame.open.iloc[first:], atol=0, rtol=0)
    np.testing.assert_allclose(ledger.mark.iloc[:-1], frame.close.iloc[first:-1], atol=0, rtol=0)
    require(ledger.shares.iloc[-1] == 0 and ledger.mark.iloc[-1] == frame.open.iloc[-1], '终点未按开盘结算')
    filled = ledger.filled_quantity.ne(0)
    qty, op = ledger.loc[filled, 'filled_quantity'].to_numpy(), ledger.loc[filled, 'open'].to_numpy()
    require((qty % cfg['lot'] == 0).all(), '非整手成交')
    units = op * (1 + np.sign(qty) * cost['slippage']) / cfg['tick']
    px = np.where(qty > 0, np.ceil(units-1e-10), np.floor(units+1e-10)) * cfg['tick']
    np.testing.assert_allclose(ledger.loc[filled, 'fill_price'], px, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ledger.loc[filled, 'commission'], np.maximum(abs(qty)*px*cost['commission'], cost['minimum']), atol=1e-8, rtol=0)
    np.testing.assert_allclose(ledger.loc[filled, 'slippage_cost'], abs(qty)*abs(px-op), atol=1e-8, rtol=0)
    return int(filled.sum())


def check_reference(cfg, data, dividends, result):
    frame = data[data.date.le(cfg['earlier_terminal'])]
    first = int(np.flatnonzero(frame.date.ge(cfg['earlier_start']))[0])
    indices = np.arange(first-1, len(frame)-1)
    gross = (frame.close + frame.dividend) / frame.previous_close
    wealth = gross.fillna(1).cumprod()
    z = ((wealth-wealth.rolling(20).mean())/wealth.rolling(20).std(ddof=1).replace(0, np.nan)).fillna(0)
    np.testing.assert_allclose(frame.wealth, wealth, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(frame.z20, z, atol=1e-9, rtol=1e-9)
    entry = ((z < -1.5) & (wealth > wealth.shift(1)) & frame.feature_valid).to_numpy(bool)
    checks, cycles_all = [], []
    for cost_id, cost in cfg['costs'].items():
        for suffix in ['ledger.parquet', 'decisions.parquet', 'cycles.csv']:
            name = f'{AUX}_{suffix}'
            require(digest(REFERENCES/'evaluation'/cost_id/name) == digest(AUX_SOURCE/'evaluation'/cost_id/name), '旧主参考复制发生改变')
        folder = REFERENCES/'earlier_diagnostic'/cost_id
        ledger = pd.read_parquet(folder/f'{AUX}_ledger.parquet')
        decisions = pd.read_parquet(folder/f'{AUX}_decisions.parquet')
        native = pd.read_csv(folder/f'{AUX}_cycles.csv').set_index('cycle_id')
        np.testing.assert_array_equal(decisions.origin_index, indices)
        require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '参考判断日历错位')
        require(pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(ledger.date)), '参考未在下一开盘执行')
        by_date, values = ledger.set_index('date'), {}
        for cycle_id, group in ledger[ledger.shares.gt(0)].groupby('cycle_id'):
            div = group.dividend_recognized.copy()
            div.iloc[0] = 0.
            values[cycle_id] = pd.Series((group.shares*group.mark+div.cumsum()).to_numpy(), index=group.date)
        last_exit, pending = -1000000, False
        for row in decisions.itertuples():
            t = row.origin_index
            close = by_date.loc[row.origin] if row.origin in by_date.index else None
            shares, cash = (int(close.shares), float(close.cash)) if close is not None else (0, cfg['initial_capital'])
            if close is not None and close.filled_quantity > 0:
                require(close.shares_before == 0, '原参考出现追加买入')
                pending = False
            if close is not None and close.filled_quantity < 0:
                require(shares == 0, '原参考出现部分卖出')
                last_exit, pending = t, False
            if shares:
                c = native.loc[close.cycle_id]
                current_return = values[close.cycle_id].loc[row.origin] / c.entry_cost_cny - 1
                pending = pending or bool(z.iloc[t] >= 0 or current_return <= -.05 or t-c.entry_index+1 >= 10)
                qty = -shares if pending else 0
            elif entry[t] and t-last_exit >= 1:
                px = math.ceil(frame.close.iloc[t]*(1+cost['slippage'])/cfg['tick']-1e-10)*cfg['tick']
                qty = int(cash // (px*cfg['lot']))*cfg['lot']
                while qty > 0 and qty*px+max(cost['minimum'], qty*px*cost['commission']) > cash+1e-9:
                    qty -= cfg['lot']
            else:
                qty = 0
            require(row.requested_quantity == qty, '参考进入、退出、止损、持有上限或等待规则不同')
            require(row.reference_weight == (0. if qty < 0 or (not shares and not qty) else 1.), '参考目标与实际状态不符')
        prior = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]]
        old_shares = np.r_[0, ledger.shares.iloc[:-1]]
        requests = decisions.requested_quantity.to_numpy().copy()
        requests[-1] = -old_shares[-1]
        np.testing.assert_array_equal(ledger.requested_quantity, requests)
        np.testing.assert_array_equal(ledger.shares, old_shares+ledger.filled_quantity)
        np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-6, rtol=0)
        np.testing.assert_allclose(ledger.equity-prior, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, atol=1e-6, rtol=0)
        np.testing.assert_allclose(ledger.equity/prior-1, ledger.net_return, atol=1e-12, rtol=0)
        m = summarize(ledger, cfg)
        stored = next(r for r in result['earlier_diagnostics'] if r['cost'] == cost_id and r['model'] == AUX)
        for key in ['net_sharpe', 'annualized_return', 'max_drawdown', 'trade_count', 'commission', 'slippage_cost']:
            require(value_equal(m[key], stored[key]), '参考保存绩效不符')
        cycles = saved_cycles(ledger, dividends, cfg)
        require(abs(sum(c['net_profit'] for c in cycles)-(ledger.equity.iloc[-1]-cfg['initial_capital'])) < 1e-5, '参考周期未还原全账户')
        cycles_all.extend({'cost': cost_id, **c} for c in cycles)
        checks.append({'cost': cost_id, 'decisions': len(decisions), 'cycles': len(cycles),
                       'fills': prices_and_costs(ledger, frame, first, cfg, cost)})
    pd.DataFrame(checks).to_csv(OUT/'saved_reference_checks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cycles_all).to_csv(OUT/'saved_reference_cycles.csv', index=False, encoding='utf-8-sig')
    return checks, cycles_all


def main():
    require(not (OUT/'saved_verification_receipt.json').exists(), '本轮已经完成核对')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    reference = json.loads((OUT/'reference_receipt.json').read_text(encoding='utf-8'))
    require(reference['config_sha256'] == digest(CONFIG), '参考配置版本不同')
    for item in cfg['frozen_files']+reference['files']:
        require(digest(ROOT/item['path']) == item['sha256'], '冻结来源或参考文件改变')
    data = pd.read_parquet(ROOT/cfg['features'])
    dividends = normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    refs, reference_cycles = check_reference(cfg, data, dividends, result)
    checks = []

    def expected(period, cost_id, frame, start, model):
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        saved = pd.read_parquet(OUT/period/cost_id/'factors.parquet')
        parents = []
        for key in [CORE, AUX]:
            source = pd.read_parquet(PARENTS[key]/period/cost_id/f'{key}_decisions.parquet')
            np.testing.assert_array_equal(source.origin_index, indices)
            require(pd.DatetimeIndex(source.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源判断日期错位')
            require(pd.DatetimeIndex(source.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源执行日期错位')
            target = np.full(len(frame), np.nan)
            target[indices] = source.reference_weight
            np.testing.assert_allclose(saved[key+'_parent_target'], target, atol=0, rtol=0, equal_nan=True)
            parents.append(target)
        weight = {'MEAN_REBOUND_AUX_25': .25, 'MEAN_REBOUND_AUX_50': .5}[model]
        require(cfg['auxiliary_weights'][model] == weight, '辅助权重不同')
        target = np.minimum(1., parents[0]+weight*parents[1])
        np.testing.assert_allclose(saved[model+'_target'], target, atol=1e-12, rtol=0, equal_nan=True)
        ledger = pd.read_parquet(OUT/period/cost_id/f'{model}_ledger.parquet')
        for key, (source, _) in CONTROLS.items():
            original = pd.read_parquet(source/period/cost_id/f'{key}_ledger.parquet')
            copied = pd.read_parquet(OUT/period/cost_id/f'{key}_ledger.parquet')
            pd.testing.assert_frame_equal(original, copied)
        checks.append({'model': model, 'period': period, 'cost': cost_id, 'decisions': len(indices),
                       'fills': prices_and_costs(ledger, frame, first, cfg, cfg['costs'][cost_id])})
        return target

    accounts, cycles, differences, count = [], [], [], 0
    for model in cfg['candidate_models']:
        def selected(period, cost_id, frame, start):
            return expected(period, cost_id, frame, start, model)
        a, c, d, n = verify_saved_target_accounts(OUT, {**cfg, 'primary': model}, result, data, dividends,
                                               selected, comparison_models=list(CONTROLS))
        accounts.extend({'model': model, **r} for r in a)
        cycles.extend({'model': model, **r} for r in c)
        differences.extend({'model': model, **r} for r in d)
        count += n
    for filename, rows in [('saved_account_checks.csv', accounts), ('saved_actual_cycles.csv', cycles),
                           ('saved_comparison_differences.csv', differences), ('saved_target_checks.csv', checks)]:
        pd.DataFrame(rows).to_csv(OUT/filename, index=False, encoding='utf-8-sig')
    require(len(accounts) == 8 and count == 11292, '新组合核对数量不同')
    receipt = {'verified_at': now(), 'status': 'PASS_EIGHT_COMPOSITES_AND_TWO_NEW_EARLIER_REFERENCES',
        'actual_accounts': 8, 'new_reference_accounts_checked': 2, 'reused_main_references_checked': 2,
        'actual_decisions_checked': count, 'reference_decisions_checked': sum(r['decisions'] for r in refs),
        'complete_actual_cycles': len(cycles), 'complete_reference_cycles': len(reference_cycles),
        'simulated_fills_checked': sum(r['fills'] for r in checks), 'reference_fills_checked': sum(r['fills'] for r in refs),
        'new_models_or_accounts': 0, 'independent_performance_validation': False,
        'security_audit_performed': False, 'reviewer_source_sha256': digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json', receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
