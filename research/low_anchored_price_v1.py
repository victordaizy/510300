"""以当时已知的新低固定量价起算日，进入后锁定该起点检查退出。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


class WeightedPriceInputs:
    def __init__(self, data):
        self.data = data
        self.wealth = data.wealth.to_numpy(float)
        self.volume = data.volume.to_numpy(float)
        high, low, close = (data[k].to_numpy(float) for k in ["high", "low", "close"])
        self.valid = np.isfinite(self.wealth) & (self.wealth > 0) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        self.valid &= (low > 0) & (low <= close) & (close <= high) & np.isfinite(self.volume) & (self.volume >= 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.typical_wealth_price = (high + low + close) / 3. * self.wealth / close
        self.valid &= np.isfinite(self.typical_wealth_price) & (self.typical_wealth_price > 0)
        self.invalid_prefix = np.r_[0, np.cumsum(~self.valid)]

    def average(self, anchor, t):
        require(0 <= anchor <= t < len(self.data), "量价起算区间不成立")
        invalid = int(self.invalid_prefix[t + 1] - self.invalid_prefix[anchor])
        if invalid:
            return {"weighted_price": None, "cumulative_volume": None, "cumulative_price_volume": None,
                    "invalid_rows": invalid, "range_status": "NO_VIEW_起算以来量价不完整"}
        volume = self.volume[anchor:t + 1]
        denominator = float(volume.sum())
        numerator = float(np.dot(self.typical_wealth_price[anchor:t + 1], volume))
        valid = denominator > 0 and np.isfinite(denominator) and np.isfinite(numerator)
        return {"weighted_price": numerator / denominator if valid else None,
                "cumulative_volume": denominator, "cumulative_price_volume": numerator, "invalid_rows": 0,
                "range_status": "量价完整" if valid else "NO_VIEW_累计成交量为零或非有限值"}


def build_factors(data, window=252):
    require(window >= 2, "新低窗口至少需要两个交易日")
    inputs = WeightedPriceInputs(data)
    wealth = data.wealth.where(np.isfinite(data.wealth) & data.wealth.gt(0))
    prior_low = wealth.shift(1).rolling(window - 1, min_periods=window - 1).min()
    rows, anchor = [], None
    for t in range(len(data)):
        new_anchor = bool(inputs.valid[t] and np.isfinite(prior_low.iloc[t]) and inputs.wealth[t] < prior_low.iloc[t])
        if new_anchor:
            anchor = t
        row = {"date": data.date.iloc[t], "wealth": inputs.wealth[t], "typical_wealth_price": inputs.typical_wealth_price[t] if inputs.valid[t] else None,
               "volume": inputs.volume[t], "daily_inputs_complete": bool(inputs.valid[t]), "prior_window_low": prior_low.iloc[t],
               "new_anchor": new_anchor, "anchor_index": anchor, "anchor_date": data.date.iloc[anchor] if anchor is not None else pd.NaT,
               "entry_cross": False, "raw_entry": 0}
        values = inputs.average(anchor, t) if anchor is not None else {"weighted_price": None, "cumulative_volume": None,
                   "cumulative_price_volume": None, "invalid_rows": None, "range_status": "NO_VIEW_尚未观察到合格的新低起点"}
        row.update(values)
        row["relative_to_weighted_price"] = inputs.wealth[t] / values["weighted_price"] - 1 if values["weighted_price"] is not None else None
        if t > 0 and anchor is not None and anchor < t and rows[-1]["anchor_index"] == anchor:
            previous = rows[-1]
            valid_pair = values["weighted_price"] is not None and previous["weighted_price"] is not None
            cross = bool(valid_pair and previous["wealth"] <= previous["weighted_price"] and inputs.wealth[t] > values["weighted_price"])
            row["entry_cross"] = cross
            row["raw_entry"] = int(cross and bool(data.feature_valid.iloc[t])) if pd.notna(data.feature_valid.iloc[t]) else 0
        rows.append(row)
    return pd.DataFrame(rows), inputs


class LockedAnchorExitController:
    def __init__(self, data, inputs, factors):
        self.data, self.inputs, self.factors = data, inputs, factors
        self.cycle_id, self.anchor = None, None

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            require(t == cycle["entry_index"], "量价起点必须从实际买入日锁定")
            origin = cycle["entry_index"] - 1
            row = self.factors.iloc[origin]
            require(row.raw_entry == 1 and pd.notna(row.anchor_index), "实际买入前没有合格量价穿越信号")
            self.cycle_id, self.anchor = cycle["cycle_id"], int(row.anchor_index)
            require(self.anchor < origin < t, "买入起点或穿越时间使用了未来信息")
        values = self.inputs.average(self.anchor, t)
        valid = values["weighted_price"] is not None
        below = bool(valid and self.inputs.wealth[t] < values["weighted_price"])
        current_anchor = self.factors.anchor_index.iloc[t]
        return {"anchor_cycle_id": cycle["cycle_id"], "locked_anchor_index": self.anchor, "locked_anchor_date": self.data.date.iloc[self.anchor],
                "current_market_anchor_index": current_anchor, "current_market_anchor_differs": pd.notna(current_anchor) and current_anchor != self.anchor,
                "locked_weighted_price": values["weighted_price"], "locked_cumulative_volume": values["cumulative_volume"],
                "locked_cumulative_price_volume": values["cumulative_price_volume"], "locked_invalid_rows": values["invalid_rows"],
                "locked_range_status": values["range_status"], "current_wealth": self.inputs.wealth[t],
                "distance_to_locked_price": self.inputs.wealth[t] / values["weighted_price"] - 1 if valid else None,
                "below_locked_price": below, "additional_exit_requested": below,
                "additional_exit_reason": "收盘跌破本次进入时固定起点的累计成交量加权价格，请求全部退出" if below else ""}
