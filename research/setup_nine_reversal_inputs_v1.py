"""四日经济价格方向的连续九次事件及固定反弹进出场。"""
from fractions import Fraction
from math import prod
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def economic_gross(close, previous, dividend):
    """用精确分数保持除息中性与完整四日价格比较。"""
    return (Fraction(str(close))+Fraction(str(dividend)))/Fraction(str(previous))


def setup_nine_factors(data, cfg):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 1 and dates.is_monotonic_increasing and not dates.has_duplicates, "九次序列必须保留完整递增日历")
    require(pd.isna(data.previous_close.iloc[0]), "九次序列首行必须是原行情结构性起点")
    require(cfg["comparison_days"] == 4 and cfg["event_count"] == 9, "只允许固定四日比较与九次首次事件")
    history, rows, failed, down, up = [], [], None, 0, 0
    for t, market in enumerate(data.itertuples()):
        values = [market.close, market.dividend]
        valid = np.isfinite(values).all() and market.close > 0 and market.dividend >= 0
        if t:
            valid = valid and np.isfinite(market.previous_close) and market.previous_close > 0
        if not valid and failed is None:
            failed = "NO_VIEW_SOURCE_GAP_REMAINDER"
        change, direction, enter, leave = np.nan, np.nan, None, None
        if failed is None and t:
            history.append(economic_gross(market.close, market.previous_close, market.dividend))
            if len(history) > 4:
                history.pop(0)
        if failed is None and t >= 4:
            gross = prod(history)
            change = float(gross-1)
            if not np.isfinite(change):
                failed = "NO_VIEW_NUMERICAL_REMAINDER"
            else:
                direction = int(gross > 1)-int(gross < 1)
                down = down+1 if direction < 0 else 0
                up = up+1 if direction > 0 else 0
                enter, leave = down == 9, up == 9
        status = failed or ("INDICATOR_AVAILABLE" if t >= 4 else "NO_VIEW_WARMUP")
        available = status == "INDICATOR_AVAILABLE"
        rows.append({"date": dates[t], "origin_index": t, "four_day_economic_return": change,
            "direction": direction, "down_count": down if available else np.nan, "up_count": up if available else np.nan,
            "entry_event": enter if available else None, "exit_event": leave if available else None, "factor_status": status})
    frame = pd.DataFrame(rows)
    return frame, {"status": failed or "INDICATORS_COMPLETE", "calendar_rows": len(frame),
        "available_indicator_days": int(frame.factor_status.eq("INDICATOR_AVAILABLE").sum()),
        "entry_events": int(frame.entry_event.eq(True).sum()), "exit_events": int(frame.exit_event.eq(True).sum()),
        "comparison_days": 4, "event_count": 9, "new_models_fit": 0}


class SetupNineController:
    def __init__(self, factors, cfg=None):
        self.down = factors.down_count.to_numpy(float)
        self.up = factors.up_count.to_numpy(float)
        self.status = factors.factor_status.to_numpy()

    def __call__(self, t, mode):
        down, up = float(self.down[t]), float(self.up[t])
        detail = {"account_mode": mode, "current_down_count": down, "current_up_count": up,
            "factor_status": str(self.status[t]), "policy_action": None}
        if mode == 2:
            return {**detail, "policy_action": 0, "decision_status": "LOCKED_EXIT_CONTINUES", "signal_reason": "此前退出受阻，继续全部卖出"}
        if self.status[t] != "INDICATOR_AVAILABLE" or not np.isfinite([down, up]).all():
            return {**detail, "decision_status": "NO_VIEW_INCOMPLETE_SETUP", "signal_reason": "连续价格比较资料不足，保持实际状态"}
        if mode == 0:
            enter = down == 9
            return {**detail, "policy_action": int(enter), "decision_status": "NINTH_WEAKNESS_ENTRY" if enter else "WAIT_NEW_NINTH_WEAKNESS",
                "signal_reason": "弱势连续段首次到九，下一开盘进入" if enter else "没有新的弱势九次事件，空仓等待"}
        leave = up == 9
        return {**detail, "policy_action": int(not leave), "decision_status": "NINTH_STRENGTH_EXIT" if leave else "HOLD_EXISTING_SHARES",
            "signal_reason": "强势连续段首次到九，下一开盘全部退出" if leave else "没有新退出事件，保持实际份额"}
