"""独立读取既有二十日上涨广度，保留来源缺失及明确的再进入条件。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def build_factors(data, source):
    required = ["date", "breadth20", "return20_scoreable_member_count", "return20_coverage_ratio",
                "point_in_time_member_count", "four_state_daily_coverage_state", "internal_feature_state"]
    require(set(required).issubset(source.columns), "缓存缺少广度和来源覆盖字段")
    source = source[required].copy()
    source["date"] = pd.to_datetime(source.date)
    require(not source.date.duplicated().any(), "广度来源日期重复")
    require(data.date.is_monotonic_increasing and not data.date.duplicated().any(), "账户市场日历必须唯一递增")
    frame = data[["date", "wealth", "feature_valid"]].merge(source, on="date", how="left", validate="one_to_one")
    numeric = ["breadth20", "return20_scoreable_member_count", "return20_coverage_ratio", "point_in_time_member_count"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ratio = frame.return20_scoreable_member_count / frame.point_in_time_member_count
    frame["breadth_valid"] = (np.isfinite(frame.breadth20) & frame.breadth20.between(0, 1)
        & frame.point_in_time_member_count.eq(300) & frame.return20_scoreable_member_count.between(294, 300)
        & frame.return20_scoreable_member_count.mod(1).eq(0) & frame.return20_coverage_ratio.ge(.98)
        & np.isclose(ratio, frame.return20_coverage_ratio, rtol=0, atol=1e-12)
        & frame.four_state_daily_coverage_state.eq("VIEW_ALLOWED"))
    frame["breadth_state"] = np.where(frame.breadth_valid, "VIEW_ALLOWED", "NO_VIEW_原广度或成员覆盖不可用")
    wealth = frame.wealth.where(np.isfinite(frame.wealth) & frame.wealth.gt(0))
    frame["wealth_mean20"] = wealth.rolling(20, min_periods=20).mean()
    frame["price_valid"] = wealth.notna() & frame.wealth_mean20.notna()
    frame["price_above20"] = frame.price_valid & wealth.gt(frame.wealth_mean20)
    frame["price_weak"] = frame.price_valid & wealth.le(frame.wealth_mean20)
    frame["breadth_majority"] = frame.breadth_valid & frame.breadth20.gt(.5)
    frame["breadth_weak"] = frame.breadth_valid & frame.breadth20.le(.5)
    frame["combined_entry_view"] = frame.breadth_valid & frame.price_valid & frame.feature_valid.eq(True)
    frame["combined_entry"] = (frame.combined_entry_view & frame.price_above20 & frame.breadth_majority).astype(int)
    frame["combined_rearm_allowed"] = frame.price_weak | frame.breadth_weak
    frame["price_entry_view"] = frame.price_valid & frame.feature_valid.eq(True)
    frame["price_entry"] = (frame.price_entry_view & frame.price_above20).astype(int)
    return frame


def rule_from_factors(factors, use_breadth=True):
    prefix = "combined" if use_breadth else "price"
    rearm = factors.combined_rearm_allowed if use_breadth else factors.price_weak
    return {"entry": factors[f"{prefix}_entry"].to_numpy(int),
            "entry_view": factors[f"{prefix}_entry_view"].to_numpy(bool),
            "rearm_allowed": rearm.to_numpy(bool), "exit": {1: np.zeros(len(factors), bool)}}


class MajorityExitController:
    def __init__(self, factors, use_breadth=True):
        self.factors, self.use_breadth = factors, use_breadth

    def __call__(self, t, cycle, current_value, peak_value):
        row = self.factors.iloc[t]
        reasons = []
        if bool(row.price_weak):
            reasons.append("财富收盘不高于含当天的二十日平均值")
        if self.use_breadth and bool(row.breadth_weak):
            reasons.append("有效成分股二十日上涨比例不高于一半")
        return {"additional_exit_requested": bool(reasons), "additional_exit_reason": "；".join(reasons),
                "price_exit_requested": bool(row.price_weak),
                "breadth_exit_requested": bool(self.use_breadth and row.breadth_weak)}
