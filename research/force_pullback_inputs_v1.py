"""分红一致的长短量价力度及固定回调进入、反弹转弱退出。"""
from decimal import Decimal
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def economic_force(close, previous, dividend, volume):
    """先按保存十进制数计算价差，避免正常除息留下虚假的正负残差。"""
    change = Decimal(str(close))-Decimal(str(previous))+Decimal(str(dividend))
    return float(change), float(change*Decimal(str(volume)))


def force_factors(data, cfg):
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and len(data) > 1, "力度日历必须完整严格递增")
    require(pd.isna(data.previous_close.iloc[0]), "力度首行应为无前收盘的结构性起点")
    periods = [cfg["fast_period"], cfg["slow_period"]]
    require(periods == [2, 13], "本轮只允许固定两期及十三期")
    rows, initial, means, count, failed = [], [], [np.nan, np.nan], 0, None
    for t, market in enumerate(data.itertuples()):
        change, raw = np.nan, np.nan
        if t == 0:
            status = "NO_VIEW_INITIAL_PREVIOUS_CLOSE"
        else:
            values = [market.close, market.previous_close, market.dividend, market.volume]
            valid = np.isfinite(values).all() and min(market.close, market.previous_close) > 0 and market.volume >= 0
            if not valid and failed is None:
                failed = "NO_VIEW_SOURCE_GAP_REMAINDER"
            if valid:
                change, raw = economic_force(*values)
            if failed is None and not np.isfinite(raw):
                failed = "NO_VIEW_NUMERICAL_REMAINDER"
            if failed is None:
                count += 1
                if count <= max(periods):
                    initial.append(raw)
                for j, period in enumerate(periods):
                    if count == period:
                        means[j] = float(np.mean(initial[:period]))
                    elif count > period:
                        rate = 2/(period+1)
                        means[j] = rate*raw+(1-rate)*means[j]
                if count >= max(periods) and not np.isfinite(means).all():
                    failed = "NO_VIEW_NUMERICAL_REMAINDER"
            status = failed or ("INDICATOR_AVAILABLE" if count >= max(periods) else "NO_VIEW_WARMUP")
        rows.append({"date": dates[t], "origin_index": t, "economic_price_change": change, "raw_force": raw,
            "fast_force": means[0] if failed is None else np.nan, "slow_force": means[1] if failed is None else np.nan,
            "observations": count, "factor_status": status})
    frame = pd.DataFrame(rows)
    return frame, {"status": failed or "INDICATORS_COMPLETE", "observations": count, "calendar_rows": len(frame),
        "available_indicator_days": int(frame.factor_status.eq("INDICATOR_AVAILABLE").sum()),
        "fast_values": int(frame.fast_force.notna().sum()), "slow_values": int(frame.slow_force.notna().sum())}


class ForcePullbackController:
    def __init__(self, factors, cfg=None):
        self.fast = factors.fast_force.to_numpy(float)
        self.slow = factors.slow_force.to_numpy(float)
        self.status = factors.factor_status.to_numpy()

    def __call__(self, t, mode):
        fast, slow = float(self.fast[t]), float(self.slow[t])
        prior = float(self.fast[t-1]) if t else np.nan
        detail = {"account_mode": mode, "current_fast_force": fast, "previous_fast_force": prior,
            "current_slow_force": slow, "factor_status": str(self.status[t]), "policy_action": None}
        if mode == 2:
            return {**detail, "policy_action": 0, "decision_status": "LOCKED_EXIT_CONTINUES", "signal_reason": "此前退出受阻，继续全部卖出"}
        if not np.isfinite([fast, slow, prior]).all():
            return {**detail, "decision_status": "NO_VIEW_INCOMPLETE_FORCE", "signal_reason": "力度历史不足，保留实际状态"}
        if mode == 0:
            enter = slow > 0 and fast < 0 and fast > prior
            return {**detail, "policy_action": int(enter), "decision_status": "PULLBACK_RECOVERING_ENTRY" if enter else "CASH_WAIT",
                "signal_reason": "长力度为正且负向短力度回升，下一开盘进入" if enter else "进入条件未同时成立，空仓等待"}
        if slow <= 0:
            return {**detail, "policy_action": 0, "decision_status": "SLOW_FORCE_NONPOSITIVE_EXIT", "signal_reason": "长力度不再为正，下一开盘退出"}
        if fast > 0 and fast < prior:
            return {**detail, "policy_action": 0, "decision_status": "POSITIVE_FAST_FORCE_FADING_EXIT", "signal_reason": "正向短力度转弱，下一开盘退出"}
        return {**detail, "policy_action": 1, "decision_status": "HOLD_EXISTING_SHARES", "signal_reason": "没有退出条件，保持已有份额"}
