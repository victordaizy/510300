"""用当时短长波动选择一个专家，不把缺失状态归入正常行情。"""
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def routing_frame(data, states, multiplier=1.5):
    require(np.isfinite(multiplier) and multiplier > 0, "波动比例必须为正")
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(states.date)), "价格与专家状态日期不一致")
    a, b = states.panic_state.to_numpy(float), states.learned_state.to_numpy(float)
    for values in [a, b]:
        require((np.isnan(values) | (values == 0) | (values == 1)).all(), "专家状态必须为零一或明确缺失")
    short, long = data.vol5.to_numpy(float), data.vol60.to_numpy(float)
    for values in [short, long]:
        require((np.isnan(values) | (np.isfinite(values) & (values >= 0))).all(), "波动率不能为负或无穷大")
    known = np.isfinite(short) & np.isfinite(long) & (long > 0)
    shock = known & (short > multiplier * long)
    selected = np.where(shock, a, b)
    target = np.where(known & np.isfinite(selected), selected, np.nan)
    ratio = np.divide(short, long, out=np.full(len(short), np.nan), where=known)
    return pd.DataFrame({"date": data.date.to_numpy(), "panic_state": a, "learned_state": b,
        "original_half_target": states.target.to_numpy(float), "vol5": short, "vol60": long,
        "short_long_ratio": ratio, "volatility_view": known,
        "regime": np.where(known, np.where(shock, "HIGH_SHORT_VOL", "NORMAL_VOL"), "NO_VIEW"),
        "selected_expert": np.where(known, np.where(shock, "PANIC", "LEARNED"), "NO_VIEW"),
        "routed_target": target})
