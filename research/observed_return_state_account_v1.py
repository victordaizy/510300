"""复用原开盘账户循环，每个收盘按实际现金和份额计算进入及退出价值。"""
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account, execute_order, require


def simulate_observed_state_account(data, dividends, config, cost, start, model_id, controller):
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    open_prices, close_prices = data.open.to_numpy(float), data.close.to_numpy(float)
    previous_close, ex_amount = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    record_events, ex_events, pay_events = {}, {}, {}
    for k, event in enumerate(dividends.itertuples()):
        for field, mapping in (("record_date", record_events), ("ex_date", ex_events)):
            event_date = getattr(event, field)
            if event_date in dates:
                mapping.setdefault(dates.get_loc(event_date), []).append((k, event.cash_dividend_per_share))
        pay_index = dates.searchsorted(event.payment_date)
        if pay_index < len(dates):
            pay_events.setdefault(pay_index, []).append((k, event.payment_date == dates[pay_index]))
    account = Account(config["initial_capital"])
    records, decisions = [], []
    previous_nav = config["initial_capital"]
    previous_mark = close_prices[anchor]

    def decision_at(t):
        result = controller(t, account)
        require(result["requested_quantity"] % config["lot"] == 0, "观察状态请求不符合整手")
        decisions.append({"origin": dates[t], "execution_date": dates[t+1] if t+1 < len(dates) else pd.NaT,
                          "origin_index": t, "simulation_only": True, **result})
        return result

    pending = decision_at(anchor)
    for day in range(first, last + 1):
        date, op, cl = dates[day], open_prices[day], close_prices[day]
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for k, amount in ex_events.get(day, []):
            value = account.entitlements.get(k, 0) * amount
            account.receivables[k] = value
            recognized += value
        for k, same_day in pay_events.get(day, []):
            if not same_day:
                value = account.receivables.pop(k, 0.0)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade_nav = account.value(op)
        execution = execute_order(account, quantity, op, previous_close[day], ex_amount[day], day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for k, same_day in pay_events.get(day, []):
                if same_day:
                    value = account.receivables.pop(k, 0.0)
                    paid += value
                    account.cash += value
            for k, amount in record_events.get(day, []):
                account.entitlements[k] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"{model_id} 在 {date} 财富恒等式不成立：{error}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade_nav, **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decision_at(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)
