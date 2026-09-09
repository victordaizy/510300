"""每个合格原点只尝试一次买入，保存原自然退出路径及其可用时间。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import Account, affordable_quantity, execute_order, fill_price, require
from research.learned_cycle_exit_v1 import FEATURES, state_values


class PathInputs:
    def __init__(self, data, dividends):
        self.data, self.dates = data, pd.DatetimeIndex(data.date)
        self.values = {k: data[k].to_numpy(float) for k in ["open", "close", "previous_close", "dividend"]}
        self.records, self.ex_dates, self.pay_dates = {}, {}, {}
        for key, event in enumerate(dividends.itertuples()):
            for field, mapping in [("record_date", self.records), ("ex_date", self.ex_dates)]:
                date = getattr(event, field)
                if date in self.dates:
                    mapping.setdefault(self.dates.get_loc(date), []).append((key, event.cash_dividend_per_share))
            day = self.dates.searchsorted(event.payment_date)
            if day < len(self.dates):
                self.pay_dates.setdefault(day, []).append((key, event.payment_date == self.dates[day]))


def signal_episodes(entry, available):
    """连续有效入场条件组成一组，失去完整数据不能当作已确认信号结束。"""
    require(len(entry) == len(available), "入场信号和有效性标记长度不一致")
    rows, current, number = [], None, 0
    for t, (on, valid) in enumerate(zip(entry, available)):
        if valid and on:
            if current is None:
                number += 1
                current = {"episode_id": number, "start_index": t, "end_index": t, "closed_index": None, "close_status": "信号尚未结束"}
            current["end_index"] = t
        elif current is not None:
            current["closed_index"] = t if valid else None
            current["close_status"] = "已经观察到原条件不成立" if valid else "NO_VIEW_数据中断，不确认信号自然结束"
            rows.append(current)
            current = None
    if current is not None:
        rows.append(current)
    return rows


def mature_episodes(episodes, paths, reference_first):
    """信号已结束且该组每个已登记进入尝试都处理完成后，整组才可用于训练。"""
    rows = []
    for event in episodes:
        local = paths[paths.episode_id.eq(event["episode_id"])]
        if local.empty:
            continue
        resolved = local.resolution_index.notna().all()
        closed = event["closed_index"] is not None
        mature = max(int(event["closed_index"]), int(local.resolution_index.max())) if closed and resolved else None
        rows.append({**event, "reference_left_truncated": event["start_index"] < reference_first,
                     "attempted_paths": len(local), "filled_paths": int(local.entry_index.notna().sum()),
                     "naturally_completed_paths": int(local.natural_exit.sum()), "unresolved_paths": int(local.resolution_index.isna().sum()),
                     "group_mature_index": mature,
                     "maturity_status": "整组自然成熟" if mature is not None else "NO_VIEW_组内尚有未结束路径或信号结束未确认"})
    return pd.DataFrame(rows)


def single_entry_path(inputs, origin, config, cost, exit_flag, mode_spec):
    """复用成交和分红原语，自然退出后停止；不生成退出之后无交易的空现金尾部。"""
    data, dates, arrays = inputs.data, inputs.dates, inputs.values
    require(0 <= origin < len(data) - 1, "假想进入原点没有下一开盘")
    account = Account(config["initial_capital"])
    first, last = origin + 1, len(data) - 1
    pending_buy = affordable_quantity(account.cash, fill_price(arrays["close"][origin], 1, cost, config["tick"]), cost, config["lot"])
    cycle, pending_reasons, cycle_dividend, peak = None, [], 0., np.nan
    previous_nav, previous_mark = config["initial_capital"], arrays["close"][origin]
    ledger, states = [], []
    result = {"path_id": origin, "entry_origin_index": origin, "entry_origin": dates[origin], "planned_entry_date": dates[first],
              "status": "尚未处理", "entry_index": None, "resolution_index": None, "natural_exit": False}
    for day in range(first, last + 1):
        old_shares, recognized, paid = account.shares, 0., 0.
        for key, amount in inputs.ex_dates.get(day, []):
            value = account.entitlements.get(key, 0) * amount
            account.receivables[key] = value
            recognized += value
        if old_shares:
            cycle_dividend += recognized
        for key, same in inputs.pay_dates.get(day, []):
            if not same:
                value = account.receivables.pop(key, 0.)
                account.cash += value
                paid += value
        terminal = day == last
        quantity = -account.shares if terminal else pending_buy if day == first else -account.shares if pending_reasons else 0
        execution = execute_order(account, quantity, arrays["open"][day], arrays["previous_close"][day], arrays["dividend"][day], day, cost, config)
        mark = arrays["open"][day] if terminal else arrays["close"][day]
        if old_shares == 0 and account.shares > 0:
            cycle = {"cycle_id": origin, "entry_index": day, "entry_date": dates[day], "mode": 1,
                     "entry_quantity": account.shares, "entry_cost_cny": execution["notional"] + execution["commission"]}
            peak = cycle["entry_cost_cny"]
            result.update({k: v for k, v in cycle.items() if k != "cycle_id"})
        elif old_shares > 0 and account.shares == 0:
            result.update(exit_index=day, exit_date=dates[day], exit_reasons="研究终点统一开盘退出" if terminal else "；".join(pending_reasons),
                          dividend_cny=cycle_dividend, holding_intervals=day - cycle["entry_index"],
                          net_profit_cny=execution["notional"] - execution["commission"] + cycle_dividend - cycle["entry_cost_cny"],
                          natural_exit=not terminal, status="自然退出已成交" if not terminal else "CENSORED_研究终点强制退出",
                          resolution_index=day if not terminal else None)
        if not terminal:
            for key, same in inputs.pay_dates.get(day, []):
                if same:
                    value = account.receivables.pop(key, 0.)
                    account.cash += value
                    paid += value
            for key, amount in inputs.records.get(day, []):
                account.entitlements[key] = account.shares
        if account.shares:
            peak = max(peak, account.shares * mark + cycle_dividend)
        nav = account.value(mark)
        price_pnl = old_shares * (arrays["open"][day] - previous_mark) + account.shares * (mark - arrays["open"][day])
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "假想单次路径财富恒等式不成立")
        account.assert_valid()
        ledger.append({"path_id": origin, "date": dates[day], "day_index": day, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                       "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
                       "net_return": nav / previous_nav - 1, "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                       "accounting_error": error, **execution})
        previous_nav, previous_mark = nav, mark
        if day == first and not account.shares:
            result.update(status="CENSORED_下一开盘为资料终点" if terminal else "首次买入未成交，不重试", entry_execution_status=execution["status"],
                          resolution_index=None if terminal else day)
            break
        if old_shares > 0 and account.shares == 0:
            break
        if terminal:
            result.update(status="CENSORED_资料终点仍未自然退出")
            break
        current_value = account.shares * arrays["close"][day] + cycle_dividend
        cycle_return = current_value / cycle["entry_cost_cny"] - 1
        if not pending_reasons:
            if exit_flag[day]:
                pending_reasons.append("本持仓模式的价格退出条件成立")
            if mode_spec["loss"] is not None and cycle_return <= -mode_spec["loss"]:
                pending_reasons.append("持仓含分红收益触及固定止损")
            if mode_spec["trail"] is not None and current_value / peak - 1 <= -mode_spec["trail"]:
                pending_reasons.append("含分红持仓价值从周期高点回落至追踪退出线")
            if mode_spec["take"] is not None and cycle_return >= mode_spec["take"]:
                pending_reasons.append("持仓含分红收益达到预定止盈")
            if mode_spec["days"] is not None and day - cycle["entry_index"] + 1 >= mode_spec["days"]:
                pending_reasons.append("预定最长持有交易日到期")
        values = state_values(data, day, cycle, current_value, peak)
        states.append({"path_id": origin, "origin_index": day, "origin": dates[day], "natural_continue": not bool(pending_reasons),
                       "pending_exit_reasons": "；".join(pending_reasons), **dict(zip(FEATURES, values))})
    result.update(last_observed_index=day, final_observed_nav=nav, remaining_shares=account.shares,
                  remaining_dividend_receivable=account.receivable(), max_accounting_error=max(abs(r["accounting_error"]) for r in ledger))
    return result, pd.DataFrame(ledger), pd.DataFrame(states)


def episode_training_rows(samples, fit_index, recent_episodes):
    """整组自然成熟后再选最近组；组内路径等权、路径内状态等权。"""
    usable = samples[samples.group_mature_index.notna() & samples.group_mature_index.le(fit_index)].copy()
    groups = usable[["episode_id", "group_mature_index"]].drop_duplicates().sort_values(["group_mature_index", "episode_id"])
    ids = groups.tail(recent_episodes).episode_id.to_list()
    chosen = usable[usable.episode_id.isin(ids)].copy().sort_values(["episode_id", "path_id", "origin_index"])
    if len(chosen):
        paths_per_group = chosen.groupby("episode_id").path_id.transform("nunique")
        states_per_path = chosen.groupby(["episode_id", "path_id"]).origin_index.transform("count")
        chosen["sample_weight"] = 1. / paths_per_group / states_per_path
        require((chosen.exit_index <= chosen.group_mature_index).all(), "整组成熟早于路径自然退出")
    else:
        chosen["sample_weight"] = pd.Series(dtype=float)
    return chosen, ids
