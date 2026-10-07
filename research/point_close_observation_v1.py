"""固定份额多头点位的完整真实日观察，文件末端不跳过实际开盘。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.point_state_reconstruction_v1 import require, open_blocked, entry_cause
from research.historic_cycles_point_translation_v1 import entitled_dividend, point_reference
from research.reference_observation_accounts_v1 import validate_observation_boundary


def candidate_signals(data, factors, model, start, next_date):
    next_date = validate_observation_boundary(data, next_date)
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(first >= 1 and pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(factors.date)), "点位来源缺少准备日或完整日历。")
    local = factors.iloc[first-1:].reset_index(drop=True)
    gate = np.where(local.drawdown60.notna(), local.drawdown60.gt(-.05).astype(float), np.nan) if model.startswith("CORE_") else local.lag_direction.to_numpy()
    return pd.DataFrame({"origin": local.date, "origin_index": local.origin_index,
                         "execution_date": pd.DatetimeIndex(data.date.iloc[first:]).append(pd.DatetimeIndex([next_date])),
                         "core_target": local.TREND_NOISE_REFERENCE_BLEND, "aux_target": local.auxiliary_raw,
                         "effective_auxiliary": local[model + "_effective_auxiliary"], "gate": gate, "target": local[model]})


def observe_points(data, dividends, signals, start):
    require(data.date.is_monotonic_increasing and data.date.is_unique, "点位价格日期不按时序。")
    require(signals.execution_date.is_unique, "点位执行日期重复。")
    daily = signals.set_index("execution_date")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(set(data.date.iloc[first:]).issubset(daily.index), "点位观察缺少某一交易日前的信号。")
    held, points, events = None, [], []
    for j in range(first, len(data)):
        date, signal = data.date.iloc[j], daily.loc[data.date.iloc[j]]
        require(signal.origin == data.date.iloc[j-1], "开盘必须使用前一实际交易日的收盘信号。")
        if not np.isfinite(signal.target):
            events.append({"date": date, "origin": signal.origin, "event": "UNKNOWN_KEEP_POSITION", "target": np.nan})
            continue
        if held is None and signal.target > 0:
            blocked = open_blocked(data, j, 1)
            if blocked:
                events.append({"date": date, "origin": signal.origin, "event": "ENTRY_BLOCKED_" + blocked, "target": signal.target})
                continue
            held = {"entry_date": date, "entry_origin": signal.origin, "entry_idx": j, "entry_raw": float(data.open.iloc[j]),
                    "entry_core_target": signal.core_target, "entry_auxiliary_target": signal.effective_auxiliary,
                    "entry_gate": signal.gate, "entry_cause": entry_cause(signal), "entry_target": signal.target}
            events.append({"date": date, "origin": signal.origin, "event": "ENTER", "target": signal.target})
        elif held is not None and signal.target == 0:
            blocked = open_blocked(data, j, -1)
            if blocked:
                events.append({"date": date, "origin": signal.origin, "event": "EXIT_BLOCKED_" + blocked, "target": signal.target})
                continue
            require(j > held["entry_idx"], "固定点位不能当日买入又卖出。")
            dividend = entitled_dividend(dividends, held["entry_date"], date)
            result = point_reference(held["entry_raw"], float(data.open.iloc[j]), dividend)
            points.append({**held, **result, "exit_date": date, "exit_origin": signal.origin, "exit_idx": j,
                           "status": "COMPLETE", "holding_sessions": j-held["entry_idx"], "exit_core_target": signal.core_target,
                           "exit_auxiliary_target": signal.effective_auxiliary, "exit_gate": signal.gate,
                           "exit_raw_auxiliary": signal.aux_target, "exit_reason": "ALL_KNOWN_COMPONENT_TARGETS_ZERO"})
            events.append({"date": date, "origin": signal.origin, "event": "EXIT", "target": signal.target})
            held = None
    if held is not None:
        date = data.date.iloc[-1]
        dividend = entitled_dividend(dividends, held["entry_date"], date)
        mark = point_reference(held["entry_raw"], float(data.close.iloc[-1]), dividend)
        points.append({**held, "status": "RIGHT_CENSORED", "exit_date": pd.NaT, "exit_origin": pd.NaT,
                       "point_net_return": np.nan, "last_observation_date": date,
                       "observed_close_mark_return": mark["point_net_return"], "holding_sessions": len(data)-1-held["entry_idx"],
                       "mark_status": "UNREALIZED_REFERENCE_ONLY_NOT_AN_EXIT"})
    columns = None if points else ["entry_date", "entry_origin", "exit_date", "exit_origin", "status", "point_net_return"]
    return pd.DataFrame(points, columns=columns), pd.DataFrame(events, columns=["date", "origin", "event", "target"])
