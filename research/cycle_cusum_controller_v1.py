"""只凭实际买入前的固定基准，逐日检查持仓收益是否累积转弱。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


class CycleCUSUMController:
    def __init__(self, data, learned_controller=None, baseline_days=60, allowance=.5, threshold=5.):
        self.data, self.learned = data, learned_controller
        self.returns = data.total_log.to_numpy(float)
        self.means = data.total_log.rolling(baseline_days, min_periods=baseline_days).mean().to_numpy(float)
        self.scales = data.total_log.rolling(baseline_days, min_periods=baseline_days).std(ddof=1).to_numpy(float)
        self.baseline_days, self.allowance, self.threshold = baseline_days, allowance, threshold
        self.cycle_id, self.last_t = None, None
        self.mean, self.scale, self.accumulated = np.nan, np.nan, None
        self.valid, self.alarmed, self.first_alarm = False, False, pd.NaT

    def __call__(self, t, cycle, current_value, peak_value):
        learned = self.learned(t, cycle, current_value, peak_value) if self.learned is not None else {}
        first_day = cycle["cycle_id"] != self.cycle_id
        if first_day:
            require(t == cycle["entry_index"], "累积转弱必须从实际买入日开始记录")
            self.cycle_id, self.last_t = cycle["cycle_id"], t
            self.anchor = cycle["entry_index"] - 1
            require(0 <= self.anchor < t, "入场前基准日期不成立")
            self.mean, self.scale = self.means[self.anchor], self.scales[self.anchor]
            self.valid = bool(self.anchor >= self.baseline_days - 1 and np.isfinite(self.mean) and np.isfinite(self.scale) and self.scale > 1e-12)
            self.accumulated = 0. if self.valid else None
            self.alarmed, self.first_alarm = False, pd.NaT
        else:
            require(t == self.last_t + 1, "持仓收盘记录不连续或被重复计算")
            self.last_t = t
        daily_return, standardized = None, None
        if not self.valid:
            status = "NO_VIEW_入场前基准或持仓收益记录不完整，本周期不补算检测"
        elif first_day:
            status = "固定入场前基准；买入当天不累积未完整持有的全天收益"
        elif not np.isfinite(self.returns[t]):
            self.valid, self.accumulated = False, None
            status = "NO_VIEW_持仓日收益缺失，本周期停止新增检测，已触发退出保留"
        else:
            daily_return = float(self.returns[t])
            standardized = float((self.mean - daily_return) / self.scale)
            self.accumulated = max(0., self.accumulated + standardized - self.allowance)
            if self.accumulated >= self.threshold:
                if not self.alarmed:
                    self.first_alarm = self.data.date.iloc[t]
                self.alarmed = True
            status = "已触发累积转弱退出" if self.alarmed else "持续观察累积转弱值"
        row = {"cusum_cycle_id": cycle["cycle_id"], "cusum_baseline_start": self.data.date.iloc[self.anchor - self.baseline_days + 1] if self.anchor >= self.baseline_days - 1 else pd.NaT,
               "cusum_baseline_end": self.data.date.iloc[self.anchor], "cusum_baseline_mean": float(self.mean) if np.isfinite(self.mean) else None,
               "cusum_baseline_sigma": float(self.scale) if np.isfinite(self.scale) else None, "cusum_daily_return": daily_return,
               "cusum_standardized_shortfall": standardized, "cusum_value": self.accumulated, "cusum_status": status,
               "cusum_information_available": self.valid, "cusum_alarm": self.alarmed, "cusum_first_alarm_origin": self.first_alarm,
               "additional_exit_requested": self.alarmed,
               "additional_exit_reason": "标准化累积转弱值达到5，请求全部退出" if self.alarmed else ""}
        return {**learned, **row}
