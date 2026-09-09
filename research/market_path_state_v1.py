"""按入场原开盘价和已确认分红建立不含已付买入费用的学习状态。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


class MarketPathTracker:
    def __init__(self, data):
        self.data = data
        self.cycle_id, self.last_index, self.entry_price, self.peak = None, None, None, None

    def observe(self, t, cycle, current_value):
        entry = int(cycle["entry_index"])
        quantity = float(cycle["entry_quantity"])
        require(entry <= t and quantity > 0 and np.isfinite(quantity), "市场路径持仓日期或份额无效")
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.last_index = cycle["cycle_id"], entry-1
            raw = float(self.data.open.iloc[entry])
            self.entry_price = raw if np.isfinite(raw) and raw > 0 else None
            self.peak = self.entry_price
        require(t > self.last_index, "市场路径收盘必须严格递增")
        if t != self.last_index+1:
            self.peak = None
        self.last_index = t
        value = float(current_value)/quantity
        if not np.isfinite(value) or value <= 0:
            self.peak = None
            value = np.nan
        if self.peak is not None:
            self.peak = max(self.peak, value)
        return {"market_entry_open": self.entry_price, "market_unit_value": value, "market_unit_peak": self.peak,
                "market_cycle_return": value/self.entry_price-1 if self.entry_price is not None else np.nan,
                "market_cycle_drawdown": value/self.peak-1 if self.peak is not None else np.nan,
                "market_path_status": "MARKET_PATH_AVAILABLE" if self.peak is not None and np.isfinite(value) else "NO_VIEW_INCOMPLETE_MARKET_PATH"}


def reference_market_states(data, ledger, decisions, cycles):
    require(cycles.cycle_id.is_unique, "参考周期编号不唯一")
    by_cycle = cycles.set_index("cycle_id")
    holding = decisions[decisions.learning_cycle_id.notna()].sort_values("origin_index")
    traces = ledger[ledger.shares.gt(0)].copy()
    require(traces.date.is_unique, "参考持仓日期重复")
    traces["cycle_dividend_cny"] = traces.groupby("cycle_id", sort=False).dividend_recognized.cumsum()
    by_date = traces.set_index("date")
    tracker, rows = MarketPathTracker(data), []
    for decision in holding.itertuples():
        cycle = by_cycle.loc[decision.learning_cycle_id].to_dict()
        cycle["cycle_id"] = int(decision.learning_cycle_id)
        account = by_date.loc[decision.origin]
        require(int(account.shares) == int(cycle["entry_quantity"]) and int(account.cycle_id) == cycle["cycle_id"], "完整参考路径的实际份额或周期不同")
        value = float(account.shares*account.mark+account.cycle_dividend_cny)
        require(abs(value/cycle["entry_cost_cny"]-1-decision.cycle_return) < 1e-12, "原参考分红状态不能从完整账本恢复")
        market = tracker.observe(int(decision.origin_index), cycle, value)
        rows.append({"cycle_id": cycle["cycle_id"], "origin_index": int(decision.origin_index), "origin": decision.origin,
                     "entry_index": int(cycle["entry_index"]), "unit_confirmed_dividend": float(account.cycle_dividend_cny/account.shares),
                     "old_cycle_return": float(decision.cycle_return), "old_cycle_drawdown": float(decision.cycle_drawdown), **market})
    result = pd.DataFrame(rows)
    require(not result.duplicated(["cycle_id", "origin_index"]).any() and len(result) == len(traces), "完整参考持仓状态存在缺行或重复")
    return result


def attach_reference_market_states(samples, states):
    columns = ["cycle_id", "origin_index", "market_cycle_return", "market_cycle_drawdown"]
    out = samples.merge(states[columns], on=["cycle_id", "origin_index"], how="left", validate="one_to_one", sort=False)
    require(len(out) == len(samples), "加入市场路径时删除或重复原训练行")
    return out
