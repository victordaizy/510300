"""每日观察状态的成熟开盘区间样本、月度分组估计和实际费用决策。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import affordable_quantity, fill_price
from research.intraday_overnight_increment_v1 import commission, require

MODES = ["OBSERVED_RETURN_STATE", "POOLED_RETURN_STATE_CONTROL"]


def label_frame(data, dividends):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "单日状态日历必须唯一递增")
    labels = []
    for i in range(1, len(data)-2):
        entry, exit_ = i+1, i+2
        r = data.total_simple.iloc[i]
        earned = dividends[dividends.record_date.ge(dates[entry]) & dividends.record_date.lt(dates[exit_])]
        maturity_date = max([dates[exit_], *earned.ex_date.to_list()])
        maturity_index = int(dates.searchsorted(maturity_date))
        op_entry, op_exit = float(data.open.iloc[entry]), float(data.open.iloc[exit_])
        known = np.isfinite(op_entry) and np.isfinite(op_exit) and op_entry > 0 and op_exit > 0
        price_return = op_exit/op_entry-1 if known else np.nan
        dividend_yield = float(earned.cash_dividend_per_share.sum())/op_entry if known else np.nan
        labels.append({"origin_index": i, "origin": dates[i], "entry_index": entry, "exit_index": exit_, "maturity_index": maturity_index,
                       "maturity_date": maturity_date, "observed_state": int(r >= 0) if np.isfinite(r) else np.nan,
                       "price_return": price_return, "new_dividend_yield": dividend_yield, "gross_total_return": price_return+dividend_yield})
    return pd.DataFrame(labels)


def estimate_schedule(data, labels, schedule, cfg):
    records = []
    for old in schedule:
        t = int(old["fit_index"])
        right, left = t-2, max(1, t-2-cfg["training_window"]+1)
        rows = labels[labels.origin_index.between(left, right)].sort_values("origin_index")
        counts = {str(state): int(rows.observed_state.eq(state).sum()) for state in [0, 1]}
        complete = len(rows) == cfg["training_window"] and np.isfinite(rows[["observed_state", "price_return", "new_dividend_yield"]].to_numpy(float)).all()
        mature = len(rows) > 0 and rows.maturity_index.le(t).all()
        supported = complete and mature and min(counts.values()) >= cfg["minimum_state_rows"]
        estimates = None
        if supported:
            estimates = {str(state): {"price_return": float(group.price_return.mean()), "new_dividend_yield": float(group.new_dividend_yield.mean())}
                         for state, group in rows.groupby("observed_state", sort=True)}
            # 明确整数键，避免状态列含缺失时浮点型字符串改变模型身份。
            estimates = {str(int(float(k))): v for k, v in estimates.items()}
            estimates["pooled"] = {"price_return": float(rows.price_return.mean()), "new_dividend_yield": float(rows.new_dividend_yield.mean())}
        status = "ESTIMATION_COMPLETE" if supported else "NO_VIEW_INCOMPLETE_WINDOW_OR_STATE_SUPPORT_OR_MATURITY"
        records.append({"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "first_origin_index": left, "last_origin_index": right,
                        "rows": len(rows), "counts": counts, "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                        "latest_maturity_index": int(rows.maturity_index.max()) if len(rows) else None, "status": status, "estimates": estimates})
    return records


def economic_values(account, close, prediction, cost, cfg):
    expected_price = close*(1+prediction["price_return"])
    require(np.isfinite(expected_price) and expected_price > 0, "条件均值给出无效预期退出价格")
    expected_dividend = close*prediction["new_dividend_yield"]
    delayed_sell_price = fill_price(expected_price, -1, cost, cfg["tick"])
    if account.shares:
        quantity = account.shares
        now_sell_price = fill_price(close, -1, cost, cfg["tick"])
        immediate = quantity*now_sell_price-commission(quantity, now_sell_price, cost)
        delayed = quantity*(delayed_sell_price+expected_dividend)-commission(quantity, delayed_sell_price, cost)
        return {"economic_quantity": quantity, "estimated_entry_gain_cny": np.nan, "estimated_continue_gain_cny": delayed-immediate,
                "expected_exit_price": expected_price, "expected_dividend_per_share": expected_dividend}
    buy_price = fill_price(close, 1, cost, cfg["tick"])
    quantity = affordable_quantity(account.cash, buy_price, cost, cfg["lot"])
    gain = np.nan
    if quantity:
        debit = quantity*buy_price+commission(quantity, buy_price, cost)
        credit = quantity*(delayed_sell_price+expected_dividend)-commission(quantity, delayed_sell_price, cost)
        gain = credit-debit
    return {"economic_quantity": quantity, "estimated_entry_gain_cny": gain, "estimated_continue_gain_cny": np.nan,
            "expected_exit_price": expected_price, "expected_dividend_per_share": expected_dividend}


class ObservedStateController:
    def __init__(self, data, records, cost, cfg, mode):
        require(mode in MODES, "观察状态方案身份不符")
        self.data, self.records, self.cost, self.cfg, self.mode = data, records, cost, cfg, mode
        self.indexes = [r["fit_index"] for r in records]
        require(self.indexes == sorted(set(self.indexes)), "观察状态月度记录不唯一递增")
        self.pending_exit = False

    def __call__(self, t, account):
        if account.shares == 0:
            self.pending_exit = False
        r = self.data.total_simple.iloc[t]
        state = int(r >= 0) if np.isfinite(r) else None
        k = bisect_right(self.indexes, t)-1
        model = self.records[k] if k >= 0 else None
        result = {"requested_quantity": 0, "reference_weight": np.nan, "action": "无新预测，保留实际份额",
                  "signal_state": "NO_VIEW_NO_MATURE_STATE_ESTIMATE", "observed_state": state, "estimate_fit_index": model["fit_index"] if model else None,
                  "predicted_price_return": np.nan, "predicted_dividend_yield": np.nan, "economic_quantity": 0,
                  "estimated_entry_gain_cny": np.nan, "estimated_continue_gain_cny": np.nan, "expected_exit_price": np.nan, "expected_dividend_per_share": np.nan}
        if model and model["status"] == "ESTIMATION_COMPLETE" and state is not None:
            require(model["latest_exit_index"] <= model["latest_maturity_index"] <= model["fit_index"] <= t, "观察状态调用未来目标或模型")
            key = str(state) if self.mode == MODES[0] else "pooled"
            prediction = model["estimates"][key]
            result.update(predicted_price_return=prediction["price_return"], predicted_dividend_yield=prediction["new_dividend_yield"], signal_state="STATE_EXPECTATION_AVAILABLE")
            values = economic_values(account, float(self.data.close.iloc[t]), prediction, self.cost, self.cfg)
            result.update(values)
            if account.shares:
                self.pending_exit = self.pending_exit or values["estimated_continue_gain_cny"] <= 0
                result.update(reference_weight=0. if self.pending_exit else 1., action="继续价值非正，下一开盘全部退出" if self.pending_exit else "继续价值为正，保留原有份额")
            elif values["economic_quantity"] and values["estimated_entry_gain_cny"] > 0:
                result.update(requested_quantity=values["economic_quantity"], reference_weight=1., action="预期单区间收益覆盖完整往返费用，下一开盘进入")
            else:
                result.update(reference_weight=0., action="预期收益未覆盖完整买卖费用或现金不足，继续等待")
        elif state is None:
            result["signal_state"] = "NO_VIEW_CURRENT_RETURN_STATE"
        if self.pending_exit and account.shares:
            result.update(requested_quantity=-account.shares, reference_weight=0., action="已有退出意图保持，下一开盘继续请求全部卖出")
        result["pending_exit_locked"] = self.pending_exit
        return result
