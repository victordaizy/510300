"""仅用已完成日线识别RSI底部失败摆动，并观察固定多头点位。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.historic_cycles_point_translation_v1 import entitled_dividend, point_reference
from research.point_state_reconstruction_v1 import open_blocked


def require(condition, message):
    if not condition:
        raise ValueError(message)


def wilder_rsi(close, period=14):
    values = np.asarray(close, dtype=float)
    require(len(values) > period and np.isfinite(values).all(), "RSI价格不足或缺失。")
    changes = np.diff(values)
    gains, losses = np.maximum(changes, 0.), np.maximum(-changes, 0.)
    result = np.full(len(values), np.nan)
    gain, loss = gains[:period].mean(), losses[:period].mean()
    for t in range(period, len(values)):
        if t > period:
            gain = ((period-1)*gain+gains[t-1])/period
            loss = ((period-1)*loss+losses[t-1])/period
        result[t] = 50. if gain == loss == 0 else 100. if loss == 0 else 100.-100./(1.+gain/loss)
    return result


def detect(dates, rsi, adjusted_low):
    """依次出现低于30、回到30上、回落守住30、突破反弹峰值，才在当天收盘确认。"""
    dates, rsi, lows = pd.DatetimeIndex(dates), np.asarray(rsi, float), np.asarray(adjusted_low, float)
    require(len(dates) == len(rsi) == len(lows), "摆动输入长度不一致。")
    mode, setup, peak, peak_index = "IDLE", None, np.nan, None
    pullback, pullback_low, pullback_rsi = None, np.nan, np.nan
    rows = []
    for t, value in enumerate(rsi):
        signal, stop, captured_setup, captured_peak, captured_pullback = False, np.nan, None, None, None
        if not np.isfinite(value) or not np.isfinite(lows[t]):
            mode, setup = "IDLE", None
        elif value < 30:
            if mode != "OVERSOLD":
                setup = t
            mode = "OVERSOLD"
        elif mode == "OVERSOLD":
            # 恰好30还不算站回30以上，等待下一根真正高于30的日线。
            if value > 30:
                mode, peak, peak_index = "REBOUND", value, t
        elif mode == "REBOUND":
            if value <= 30:
                mode, setup = "IDLE", None
            elif value >= peak:
                peak, peak_index = value, t
            else:
                mode, pullback, pullback_low, pullback_rsi = "PULLBACK", t, lows[t], value
        elif mode == "PULLBACK":
            if value <= 30:
                mode, setup = "IDLE", None
            else:
                pullback_low, pullback_rsi = min(pullback_low, lows[t]), min(pullback_rsi, value)
                if value > peak:
                    signal, stop = True, pullback_low
                    captured_setup, captured_peak, captured_pullback = setup, peak_index, pullback
                    mode, setup = "IDLE", None
        rows.append({"date": dates[t], "rsi14": value, "state_after_close": mode, "entry_signal": signal,
                     "structural_stop_index": stop, "setup_index": captured_setup,
                     "rebound_peak_index": captured_peak, "pullback_start_index": captured_pullback,
                     "rebound_peak_rsi": peak if signal else np.nan,
                     "pullback_min_rsi": pullback_rsi if signal else np.nan})
    result = pd.DataFrame(rows)
    for column in ("setup_index", "rebound_peak_index", "pullback_start_index"):
        result[column] = result[column].astype("Int64")
    return result


def observe(data, dividends, signals, start, max_holding_closes=20, reward_multiple=2.):
    """事件只在次日开盘尝试一次；止损、2R目标和时间退出均收盘确认、随后开盘执行。"""
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(signals.date)), "信号与行情日历不一致。")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(first > 0, "观察起点缺少前一收盘。")
    held, pending_exit, points, events = None, None, [], []
    for j in range(first, len(data)):
        date, previous = data.date.iloc[j], signals.iloc[j-1]
        exited = False
        if held is not None and pending_exit is not None:
            blocked = open_blocked(data, j, -1)
            if blocked:
                events.append({"date": date, "event": "EXIT_BLOCKED", "reason": blocked})
            else:
                cash = entitled_dividend(dividends, held["entry_date"], date)
                result = point_reference(held["entry_raw"], float(data.open.iloc[j]), cash)
                points.append({**held, **result, "exit_date": date, "exit_idx": j,
                               "exit_origin": pending_exit["origin"], "exit_reason": pending_exit["reason"],
                               "holding_sessions": j-held["entry_idx"], "status": "COMPLETE",
                               "realized_net_r": result["point_net_return"]/held["initial_risk_fraction"]})
                events.append({"date": date, "event": "EXIT", "reason": pending_exit["reason"]})
                held, pending_exit, exited = None, None, True
        if previous.entry_signal:
            if held is not None or exited:
                events.append({"date": date, "event": "IGNORE_SIGNAL_OCCUPIED_OR_EXIT_PRIORITY", "reason": "已有点位或当日退出优先"})
            else:
                entry_index_price = float(data.ao.iloc[j])
                stop = float(previous.structural_stop_index)
                blocked = open_blocked(data, j, 1)
                if blocked or entry_index_price <= stop:
                    events.append({"date": date, "event": "CANCEL_ENTRY", "reason": blocked or "开盘已不高于结构失效线"})
                else:
                    risk = entry_index_price-stop
                    held = {"entry_date": date, "entry_origin": signals.date.iloc[j-1], "entry_idx": j,
                            "entry_raw": float(data.open.iloc[j]), "entry_rsi14": float(previous.rsi14),
                            "structural_stop_index": stop, "planned_target_index": entry_index_price+reward_multiple*risk,
                            "initial_risk_per_share": risk, "initial_risk_fraction": risk/float(data.open.iloc[j]),
                            "planned_reward_multiple": reward_multiple}
                    events.append({"date": date, "event": "ENTER", "reason": "前收盘RSI底部失败摆动确认"})
        if held is not None and pending_exit is None:
            close = float(data.ac.iloc[j])
            reason = ("STRUCTURAL_STOP_CLOSE" if close <= held["structural_stop_index"] else
                      "PLANNED_2R_CLOSE" if close >= held["planned_target_index"] else
                      "TWENTY_CLOSE_TIME_EXIT" if j-held["entry_idx"]+1 >= max_holding_closes else None)
            if reason:
                pending_exit = {"origin": date, "reason": reason}
                events.append({"date": date, "event": "EXIT_REQUEST_AT_CLOSE", "reason": reason})
    if held is not None:
        cash = entitled_dividend(dividends, held["entry_date"], data.date.iloc[-1])
        mark = point_reference(held["entry_raw"], float(data.close.iloc[-1]), cash)
        points.append({**held, "exit_date": pd.NaT, "exit_idx": np.nan, "exit_origin": pd.NaT,
                       "status": "RIGHT_CENSORED", "point_net_return": np.nan, "realized_net_r": np.nan,
                       "last_observation_date": data.date.iloc[-1], "observed_close_mark_return": mark["point_net_return"],
                       "holding_sessions": len(data)-1-held["entry_idx"],
                       "pending_exit_reason": pending_exit["reason"] if pending_exit else None})
    columns = None if points else ["entry_date", "entry_origin", "exit_date", "exit_origin", "status", "point_net_return", "realized_net_r"]
    return pd.DataFrame(points, columns=columns), pd.DataFrame(events, columns=["date", "event", "reason"])
