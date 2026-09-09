"""月末收益风险仓位与一致的趋势进入、退出及再次进入。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, choose_order, execute_order, require


def simulate_utility_exit_account(data, dividends, config, cost, prediction, event_mask,
                                  expiry_enabled=False, trend_enabled=False, entry_trend_required=False):
    first = int(np.flatnonzero(data.date >= pd.Timestamp(config['evaluation_start']))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    op = data.open.to_numpy(float)
    cl = data.close.to_numpy(float)
    previous = data.previous_close.to_numpy(float)
    distribution = data.dividend.to_numpy(float)
    variance = data.variance60.to_numpy(float)
    trend = data.sma120.to_numpy(float)
    prediction = np.asarray(prediction, float)
    event_mask = np.asarray(event_mask, bool)
    require(len(prediction) == len(data) == len(event_mask), '预测和事件时钟长度不一致')
    record_events, ex_events, pay_events = {}, {}, {}
    for key, event in enumerate(dividends.itertuples()):
        for field, mapping in [('record_date', record_events), ('ex_date', ex_events)]:
            when = getattr(event, field)
            if when in dates:
                mapping.setdefault(dates.get_loc(when), []).append((key, event.cash_dividend_per_share))
        index = dates.searchsorted(event.payment_date)
        if index < len(dates):
            pay_events.setdefault(index, []).append((key, event.payment_date == dates[index]))
    account = Account(config['initial_capital'])
    previous_nav, previous_mark = config['initial_capital'], cl[anchor]
    latest_valid_origin = None
    pending_exit, pending_reasons = False, []
    reference = np.nan
    records, decisions = [], []

    def decide(t):
        nonlocal latest_valid_origin, pending_exit, pending_reasons, reference
        scheduled = t == anchor or bool(event_mask[t])
        value = prediction[t]
        if scheduled and np.isfinite(value):
            latest_valid_origin = t
        age = t - latest_valid_origin if latest_valid_origin is not None else None
        reasons, new_trigger = [], False
        if account.shares and not pending_exit:
            if expiry_enabled and age is not None and age >= config['forecast_validity_trading_days']:
                reasons.append('最近一次有效盈利预测已满六十交易日，且没有更新')
            broken = t > 0 and np.isfinite(trend[t-1:t+1]).all() and (trend[t-1:t+1] <= 0).all()
            if trend_enabled and broken:
                reasons.append('含分红价格连续两个收盘不高于一百二十日均线')
            if reasons:
                pending_exit, pending_reasons, new_trigger = True, reasons, True
        if account.shares and pending_exit:
            result = {'requested_quantity': -account.shares, 'reference_weight': 0.,
                      'action': '独立退出条件触发，清仓请求持续到成交', 'signal_state': 'EXIT_PENDING_UNTIL_FILLED'}
            reasons = pending_reasons
        elif scheduled and np.isfinite(value):
            result = choose_order(account, cl[t], value, config['horizon'] * variance[t], cost, config)
            result['signal_state'] = 'ORIGINAL_MONTHLY_UTILITY_VIEW'
            if entry_trend_required and result['requested_quantity'] > 0 and not (np.isfinite(trend[t]) and trend[t] > 0):
                result = {'requested_quantity': 0, 'reference_weight': account.shares * cl[t] / account.value(cl[t]),
                          'action': '趋势未恢复向上，禁止新进入或增加，已有减持及退出规则保留',
                          'signal_state': 'ENTRY_OR_ADDITION_BLOCKED_BY_KNOWN_TREND'}
        else:
            result = {'requested_quantity': 0, 'reference_weight': np.nan if scheduled else reference,
                      'action': '资料不足保留已有份额' if scheduled else '非月末维持份额并检查独立退出',
                      'signal_state': 'NO_VIEW_KEEP_EXISTING_SHARES' if scheduled else 'HOLD_BETWEEN_MONTH_ENDS'}
        reference = float(result['reference_weight'])
        row = {'origin': dates[t], 'execution_date': dates[t+1] if t+1 < len(dates) else pd.NaT,
               'origin_index': t, 'simulation_only': True, 'forecast_return_60': value,
               'latest_valid_forecast_origin': dates[latest_valid_origin] if latest_valid_origin is not None else pd.NaT,
               'forecast_age_trading_days': age, 'new_exit_trigger': new_trigger,
               'exit_reasons': '；'.join(reasons), **result}
        decisions.append(row)
        return row

    pending = decide(anchor)
    for day in range(first, last+1):
        old_shares, recognized, paid = account.shares, 0., 0.
        for key, amount in ex_events.get(day, []):
            value = account.entitlements.get(key, 0) * amount
            account.receivables[key] = value
            recognized += value
        for key, same_day in pay_events.get(day, []):
            if not same_day:
                value = account.receivables.pop(key, 0.)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending['requested_quantity'])
        before = account.value(op[day])
        execution = execute_order(account, quantity, op[day], previous[day], distribution[day], day, cost, config)
        mark = op[day] if terminal else cl[day]
        if old_shares > 0 and account.shares == 0:
            pending_exit, pending_reasons = False, []
        if not terminal:
            for key, same_day in pay_events.get(day, []):
                if same_day:
                    value = account.receivables.pop(key, 0.)
                    paid += value
                    account.cash += value
            for key, amount in record_events.get(day, []):
                account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op[day] - previous_mark) + account.shares * (mark - op[day])
        error = nav - previous_nav - price_pnl - recognized + execution['commission'] + execution['slippage_cost']
        require(abs(error) < 1e-6, '收益风险仓位与独立退出账户的净值恒等式不成立')
        account.assert_valid()
        records.append({'date': dates[day], 'open': op[day], 'mark': mark,
                        'mark_clock': 'OPEN_TERMINAL' if terminal else 'CLOSE',
                        'cash': account.cash, 'shares': account.shares, 'dividend_receivable': account.receivable(),
                        'equity': nav, 'net_return': nav / previous_nav - 1, 'pnl': nav - previous_nav,
                        'price_pnl': price_pnl, 'dividend_recognized': recognized, 'dividend_paid': paid,
                        'exposure': account.shares * mark / nav, 'accounting_error': error,
                        'origin': dates[day-1], 'terminal_unliquidated': bool(terminal and account.shares),
                        'turnover': execution['notional'] / before,
                        'decision_action': '研究终点统一开盘清仓' if terminal else pending['action'],
                        'execution_exit_reasons': '研究终点统一开盘清仓' if terminal else pending['exit_reasons'],
                        'forecast_age_trading_days': pending['forecast_age_trading_days'], **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)
