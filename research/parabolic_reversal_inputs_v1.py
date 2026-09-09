"""因果含分红高低价的抛物线转向。

递推参考TA-Lib的TA_SAR；原版权及许可随项目保存在
docs/reference/ta_lib_sar_20260908/LICENSE.txt，实际参考源码同目录。
本实现增加缺失重启、全部内部状态与中文研究所需的0或1目标；
不声称与外部编译库的融合浮点运算逐位相同。
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def parabolic_frame(data, acceleration=.02, maximum=.2):
    require(np.isfinite(acceleration) and np.isfinite(maximum) and 0 < acceleration <= maximum, "加速步长和上限必须为正且顺序正确")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "转向指标日期必须严格递增")
    raw = data[["high", "low", "close", "dividend", "wealth"]].to_numpy(float)
    require((np.isnan(raw) | np.isfinite(raw)).all(), "价格或分红不能为无穷大")
    for k in [0, 1, 2, 4]:
        require((np.isnan(raw[:, k]) | (raw[:, k] > 0)).all(), "价格及财富值必须为正")
    require((np.isnan(raw[:, 3]) | (raw[:, 3] >= 0)).all(), "分红不能为负")
    valid = np.isfinite(raw).all(axis=1)
    require((raw[valid, 1] <= raw[valid, 2]).all() and (raw[valid, 2] <= raw[valid, 0]).all(), "收盘必须处于当天高低价之间")
    scale = raw[:, 4] / (raw[:, 2] + raw[:, 3])
    high, low, close = (raw[:, 0] + raw[:, 3]) * scale, (raw[:, 1] + raw[:, 3]) * scale, raw[:, 4]
    values = {name: np.full(len(data), np.nan) for name in ["sar_input", "sar_level", "next_sar", "extreme_price", "acceleration", "trend_state", "target", "reversed", "initialized"]}
    states = np.full(len(data), "NO_VIEW_FIRST_VALID_BAR", object)
    previous_valid, direction, extreme, speed, sar = None, None, None, None, None
    for t in range(len(data)):
        if not valid[t]:
            previous_valid, direction, extreme, speed, sar = None, None, None, None, None
            states[t] = "NO_VIEW_INPUT_RESET"
            continue
        if previous_valid is None:
            previous_valid = t
            continue
        initialized = direction is None
        if initialized:
            rise = high[t] - high[previous_valid]
            fall = low[previous_valid] - low[t]
            direction = 0 if fall > 0 and fall > rise else 1
            extreme = high[t] if direction else low[t]
            sar = low[previous_valid] if direction else high[previous_valid]
            speed = acceleration
            previous_high, previous_low = high[t], low[t]
        else:
            previous_high, previous_low = high[previous_valid], low[previous_valid]
        input_sar = sar
        switched = (direction == 1 and low[t] <= sar) or (direction == 0 and high[t] >= sar)
        if switched:
            direction = 1 - direction
            sar = min(extreme, previous_low, low[t]) if direction else max(extreme, previous_high, high[t])
            extreme = high[t] if direction else low[t]
            speed = acceleration
        elif (direction == 1 and high[t] > extreme) or (direction == 0 and low[t] < extreme):
            extreme = high[t] if direction else low[t]
            speed = min(maximum, speed + acceleration)
        output_sar = sar
        candidate = math.fma(float(speed), float(extreme - sar), float(sar))
        sar = min(candidate, previous_low, low[t]) if direction else max(candidate, previous_high, high[t])
        for name, value in {"sar_input": input_sar, "sar_level": output_sar, "next_sar": sar, "extreme_price": extreme,
            "acceleration": speed, "trend_state": direction, "target": direction, "reversed": int(switched), "initialized": int(initialized)}.items():
            values[name][t] = value
        states[t] = "VIEW_UP_STATE" if direction else "VIEW_DOWN_STATE"
        previous_valid = t
    return pd.DataFrame({"date": dates, "wealth_high": high, "wealth_low": low, "wealth_close": close,
        "input_valid": valid, "source_state": states, **values})
