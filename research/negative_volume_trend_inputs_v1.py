"""只累计缩量日的经济涨跌，按固定255日均线进入和退出。"""
from decimal import Decimal
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def economic_return(close, previous, dividend):
    """十进制价差先消除正常除息，收益口径为简单涨跌幅。"""
    prior = Decimal(str(previous))
    return float((Decimal(str(close))-prior+Decimal(str(dividend)))/prior)


def negative_volume_factors(data, cfg):
    dates = pd.DatetimeIndex(data.date)
    require(len(data) > 1 and dates.is_monotonic_increasing and not dates.has_duplicates, "缩量指标必须保留严格递增日历")
    require(pd.isna(data.previous_close.iloc[0]), "缩量指标首行必须是原行情结构性起点")
    require(cfg["signal_period"] == 255 and cfg["initial_index"] == 1000., "只允许已约定的255期及初值1000")
    level, average, count, failed = 1000., np.nan, 0, None
    initial, rows = [], []
    previous_volume = np.nan
    for t, market in enumerate(data.itertuples()):
        change, selected, decreased = np.nan, np.nan, None
        values = [market.close, market.dividend, market.volume]
        valid = np.isfinite(values).all() and market.close > 0 and market.dividend >= 0 and market.volume >= 0
        if t:
            valid = valid and np.isfinite([market.previous_close, previous_volume]).all() and market.previous_close > 0 and previous_volume >= 0
        if not valid and failed is None:
            failed = "NO_VIEW_SOURCE_GAP_REMAINDER"
        if failed is None:
            if t:
                change = economic_return(market.close, market.previous_close, market.dividend)
                decreased = bool(market.volume < previous_volume)
                selected = 100.*change if decreased else 0.
                level += selected
            if not np.isfinite(level):
                failed = "NO_VIEW_NUMERICAL_REMAINDER"
            else:
                count += 1
                if count <= 255:
                    initial.append(level)
                if count == 255:
                    average = float(np.mean(initial))
                elif count > 255:
                    average += (level-average)/128.
                if count >= 255 and not np.isfinite(average):
                    failed = "NO_VIEW_NUMERICAL_REMAINDER"
        status = failed or ("INDICATOR_AVAILABLE" if count >= 255 else "NO_VIEW_WARMUP")
        rows.append({"date": dates[t], "origin_index": t, "economic_return": change,
            "volume_decreased": decreased, "selected_percentage_change": selected,
            "nvi": level if failed is None else np.nan, "signal_mean": average if failed is None else np.nan,
            "observations": count, "factor_status": status})
        previous_volume = market.volume
    frame = pd.DataFrame(rows)
    return frame, {"status": failed or "INDICATORS_COMPLETE", "calendar_rows": len(frame), "observations": count,
        "available_indicator_days": int(frame.factor_status.eq("INDICATOR_AVAILABLE").sum()),
        "volume_decrease_days": int(frame.volume_decreased.eq(True).sum()),
        "formula": "ADDITIVE_PERCENTAGE_POINTS_NOT_MULTIPLICATIVE_WEALTH", "signal_period": 255}


class NegativeVolumeController:
    def __init__(self, factors, cfg=None):
        self.level = factors.nvi.to_numpy(float)
        self.average = factors.signal_mean.to_numpy(float)
        self.status = factors.factor_status.to_numpy()

    def __call__(self, t, mode):
        level, average = float(self.level[t]), float(self.average[t])
        detail = {"account_mode": mode, "current_nvi": level, "current_signal_mean": average,
            "factor_status": str(self.status[t]), "policy_action": None}
        if mode == 2:
            return {**detail, "policy_action": 0, "decision_status": "LOCKED_EXIT_CONTINUES", "signal_reason": "此前退出受阻，继续全部卖出"}
        if self.status[t] != "INDICATOR_AVAILABLE" or not np.isfinite([level, average]).all():
            return {**detail, "decision_status": "NO_VIEW_INCOMPLETE_NEGATIVE_VOLUME", "signal_reason": "缩量指标历史不足，保留实际状态"}
        if level > average:
            return {**detail, "policy_action": 1, "decision_status": "NVI_ABOVE_MEAN_ENTRY" if mode == 0 else "NVI_ABOVE_MEAN_HOLD",
                "signal_reason": "缩量累计指标高于均线，空仓下一开盘进入，持仓保持份额"}
        if level < average:
            return {**detail, "policy_action": 0, "decision_status": "NVI_BELOW_MEAN_EXIT" if mode == 1 else "NVI_BELOW_MEAN_CASH",
                "signal_reason": "缩量累计指标低于均线，持仓下一开盘退出，空仓等待"}
        return {**detail, "policy_action": int(mode == 1), "decision_status": "NVI_EQUALS_MEAN_KEEP_STATE", "signal_reason": "指标等于均线，保持实际状态"}
