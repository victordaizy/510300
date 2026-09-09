"""用除息修正后的低开收复进入，以日内转弱退出。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recovery_frame(data: pd.DataFrame, tick: float = .001) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(data.date))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("行情日期缺失、重复或倒序")
    if dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("行情必须使用无时区的完整交易日期")
    if not np.isfinite(tick) or tick <= 0:
        raise ValueError("报价单位必须为正")
    raw = data[["open", "close", "previous_close", "dividend"]].to_numpy(float)
    known = np.isfinite(raw)
    if np.isinf(raw).any() or np.any(raw[:, :3][known[:, :3]] <= 0) or np.any(raw[:, 3][known[:, 3]] < 0):
        raise ValueError("价格非正、分红为负或金额无穷大")
    scaled = raw[known] / tick
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) > 1e15) or np.any(np.abs(scaled - np.rint(scaled)) > 1e-7):
        raise ValueError("输入金额不能按已核对报价单位精确表示，不作四舍五入修补")
    units = np.zeros(raw.shape, dtype=np.int64)
    units[known] = np.rint(scaled).astype(np.int64)
    complete = known.all(axis=1)
    opening_gap = units[:, 0] + units[:, 3] - units[:, 2]
    recovery = units[:, 1] + units[:, 3] - units[:, 2]
    intraday = units[:, 1] - units[:, 0]
    entry = complete & (opening_gap < 0) & (recovery > 0)
    weak = complete & (intraday <= 0)
    intent = None
    targets, latent, reasons = [], [], []
    for valid, enter, exit_now in zip(complete, entry, weak):
        if not valid:
            targets.append(np.nan)
            latent.append(float(intent) if intent is not None else np.nan)
            reasons.append("资料缺失，暂停调整，保留此前意图")
            continue
        if exit_now:
            intent, reason = 0., "日内转弱或持平，目标归零"
        elif enter:
            intent, reason = 1., "低开后收盘严格收复前收盘"
        elif intent is None:
            intent, reason = 0., "首次资料完整且未发生进入事件，初始化空仓"
        else:
            reason = "没有新进入或退出事件，保持此前意图"
        targets.append(intent)
        latent.append(intent)
        reasons.append(reason)
    return pd.DataFrame({"date": dates, "open": raw[:, 0], "close": raw[:, 1], "previous_close": raw[:, 2], "dividend": raw[:, 3],
        "opening_gap_after_dividend": np.where(complete, opening_gap * tick, np.nan),
        "close_recovery_after_dividend": np.where(complete, recovery * tick, np.nan),
        "intraday_change": np.where(complete, intraday * tick, np.nan),
        "entry_event": np.where(complete, entry.astype(float), np.nan),
        "weak_exit_event": np.where(complete, weak.astype(float), np.nan),
        "source_state": np.where(complete, "VIEW", "NO_VIEW_MISSING_INPUT"),
        "policy_intent": latent, "target": targets, "policy_reason": reasons})
