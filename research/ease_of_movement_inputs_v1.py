"""用中点移动、区间和实际成交量计算简易波动，并执行固定零线进出场。"""
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import affordable_quantity, fill_price
from research.intraday_overnight_increment_v1 import require

PRIMARY = "EASE_OF_MOVEMENT_ZERO_CROSS"


def factor_frame(data, window=14):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "简易波动日历必须唯一递增")
    columns = ["high", "low", "close", "dividend", "wealth", "volume"]
    finite = np.isfinite(data[columns].to_numpy(float)).all(axis=1)
    valid = pd.Series(finite, index=data.index) & data.low.gt(0) & data.high.ge(data.low) & data.close.between(data.low, data.high) & data.wealth.gt(0) & data.volume.gt(0) & (data.close+data.dividend).gt(0)
    scale = data.wealth/(data.close+data.dividend)
    high = ((data.high+data.dividend)*scale).where(valid)
    low = ((data.low+data.dividend)*scale).where(valid)
    midpoint = (high+low)/2
    movement = midpoint.diff()
    interval = high-low
    # 正常零区间贡献为零；无效或缺失成交量保持未知。
    raw = (movement*interval*100000000./data.volume).where(valid)
    smoothed = raw.rolling(window, min_periods=window).mean()
    entry = smoothed.gt(0) & smoothed.shift().le(0)
    exit_ = smoothed.le(0)
    return pd.DataFrame({"date": data.date, "wealth_high": high, "wealth_low": low, "wealth_midpoint": midpoint, "midpoint_change": movement, "wealth_range": interval,
                         "input_valid": valid, "movement_per_volume": raw, "ease_of_movement": smoothed, "entry_cross": entry, "exit_nonpositive": exit_,
                         "source_state": np.where(smoothed.notna(), "EASE_OF_MOVEMENT_AVAILABLE", "NO_VIEW_INCOMPLETE_MOVEMENT_WINDOW")})


class EaseOfMovementController:
    def __init__(self, data, factors, cost, cfg):
        require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(factors.date)), "简易波动与账户日历不符")
        self.data, self.factors, self.cost, self.cfg = data, factors, cost, cfg
        self.pending_exit = False

    def __call__(self, t, account):
        if account.shares == 0:
            self.pending_exit = False
        row = self.factors.iloc[t]
        result = {"requested_quantity": 0, "reference_weight": np.nan, "action": "指标未知，保留实际份额", "signal_state": row.source_state,
                  "ease_of_movement": row.ease_of_movement, "entry_cross": bool(row.entry_cross), "exit_nonpositive": bool(row.exit_nonpositive)}
        if np.isfinite(row.ease_of_movement):
            if account.shares:
                self.pending_exit = self.pending_exit or row.exit_nonpositive
                result.update(reference_weight=0. if self.pending_exit else 1., action="指标仍为正，保留原有份额")
            elif row.entry_cross:
                buy_price = fill_price(float(self.data.close.iloc[t]), 1, self.cost, self.cfg["tick"])
                quantity = affordable_quantity(account.cash, buy_price, self.cost, self.cfg["lot"])
                result.update(requested_quantity=quantity, reference_weight=1. if quantity else 0., action="指标由非正转正，下一开盘按实际现金进入" if quantity else "现金不足一手，等待下次转正")
            else:
                result.update(reference_weight=0., action="未出现新的零线上穿，保持等待")
        if self.pending_exit and account.shares:
            result.update(requested_quantity=-account.shares, reference_weight=0., action="退出意图保持，下一开盘全部卖出")
        result["pending_exit_locked"] = bool(self.pending_exit)
        return result
