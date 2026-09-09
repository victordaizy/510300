"""T_ONLY追加式每日事件账本及哈希链。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd


ZERO_HASH = "0" * 64


class ForwardLedgerError(ValueError):
    """前瞻事件账本不连续或既有记录被改写。"""


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


def event_hash(record: Mapping[str, Any]) -> str:
    payload = {
        str(key): _json_value(value)
        for key, value in record.items()
        if key != "event_hash"
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_event_chain(records: list[dict[str, Any]]) -> None:
    previous = ZERO_HASH
    dates: list[pd.Timestamp] = []
    for record in records:
        event_date = pd.Timestamp(record["event_date"]).normalize()
        dates.append(event_date)
        if record.get("previous_event_hash") != previous:
            raise ForwardLedgerError(f"{event_date.date()}前序哈希不一致")
        expected = event_hash(record)
        if record.get("event_hash") != expected:
            raise ForwardLedgerError(f"{event_date.date()}事件内容哈希不一致")
        previous = expected
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ForwardLedgerError("事件日期必须严格递增且唯一")


def _time(date: pd.Timestamp, clock: str, timezone: str) -> str:
    return pd.Timestamp(f"{date.date()} {clock}", tz=timezone).isoformat()


def build_daily_event_records(
    *,
    daily_ledger: pd.DataFrame,
    executions: pd.DataFrame,
    cycles: pd.DataFrame,
    calendar: pd.DataFrame,
    action_targets: Mapping[str, float],
    timezone: str = "Asia/Shanghai",
    signal_cutoff_time: str = "15:00:00",
    signal_record_time: str = "16:30:00",
    next_open_time: str = "09:30:00",
) -> list[dict[str, Any]]:
    """从冻结模拟事件生成不含收益指标的逐日哈希链。"""

    required = {
        "date",
        "mode",
        "tier",
        "shares",
        "day_commission",
        "day_slippage",
        "day_raw_notional",
        "executed_reason",
        "next_signal_reason",
    }
    missing = required - set(daily_ledger.columns)
    if missing:
        raise ForwardLedgerError(f"日账本缺少字段：{sorted(missing)}")
    frame = daily_ledger.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ForwardLedgerError("日账本存在重复日期")

    calendar_dates = pd.to_datetime(
        calendar["trade_date"], errors="raise"
    ).dt.normalize().drop_duplicates().sort_values().tolist()
    if not frame.empty:
        expected = [
            value
            for value in calendar_dates
            if frame["date"].min() <= value <= frame["date"].max()
        ]
        if frame["date"].tolist() != expected:
            raise ForwardLedgerError("前瞻日账本与冻结交易日历不连续")
    next_trade_date = {
        current: calendar_dates[index + 1]
        for index, current in enumerate(calendar_dates[:-1])
    }

    execution_frame = executions.copy()
    if not execution_frame.empty:
        execution_frame["execution_date"] = pd.to_datetime(
            execution_frame["execution_date"], errors="raise"
        ).dt.normalize()
        execution_frame["signal_date"] = pd.to_datetime(
            execution_frame["signal_date"], errors="raise"
        ).dt.normalize()
    cycle_frame = cycles.copy()
    if not cycle_frame.empty:
        cycle_frame["entry_date"] = pd.to_datetime(
            cycle_frame["entry_date"], errors="raise"
        ).dt.normalize()
        cycle_frame["exit_date"] = pd.to_datetime(
            cycle_frame["exit_date"], errors="raise"
        ).dt.normalize()

    records: list[dict[str, Any]] = []
    previous = ZERO_HASH
    for row in frame.to_dict("records"):
        date = pd.Timestamp(row["date"]).normalize()
        reason = row.get("next_signal_reason")
        reason = None if pd.isna(reason) else str(reason)
        signal = None
        if reason is not None:
            if reason not in action_targets:
                raise ForwardLedgerError(f"未注册信号动作：{reason}")
            if date not in next_trade_date:
                raise ForwardLedgerError(f"{date.date()}之后缺少下一交易日")
            signal = {
                "reason": reason,
                "target_exposure": float(action_targets[reason]),
                "data_cutoff": _time(date, signal_cutoff_time, timezone),
                "signal_time": _time(date, signal_record_time, timezone),
                "effective_time": _time(
                    next_trade_date[date], next_open_time, timezone
                ),
                "execution_rule": "NEXT_TRADING_DAY_OPEN_SHADOW_ONLY",
            }

        day_executions: list[dict[str, Any]] = []
        if not execution_frame.empty:
            for trade in execution_frame.loc[
                execution_frame["execution_date"].eq(date)
            ].to_dict("records"):
                day_executions.append(
                    {
                        "signal_date": str(pd.Timestamp(trade["signal_date"]).date()),
                        "execution_date": str(date.date()),
                        "reason": str(trade["reason"]),
                        "side": str(trade["side"]),
                        "trade_shares": int(trade["trade_shares"]),
                        "raw_notional": float(trade["raw_notional"]),
                        "commission": float(trade["commission"]),
                        "slippage_cost": float(trade["slippage_cost"]),
                        "target_exposure": float(trade["target_exposure"]),
                    }
                )

        closed_cycles: list[dict[str, Any]] = []
        if not cycle_frame.empty:
            for cycle in cycle_frame.loc[cycle_frame["exit_date"].eq(date)].to_dict(
                "records"
            ):
                closed_cycles.append(
                    {
                        "entry_date": str(pd.Timestamp(cycle["entry_date"]).date()),
                        "exit_date": str(date.date()),
                        "entry_reason": str(cycle["entry_reason"]),
                        "exit_reason": str(cycle["exit_reason"]),
                        "holding_calendar_days": int(cycle["holding_calendar_days"]),
                    }
                )

        record: dict[str, Any] = {
            "schema_version": "T_ONLY_FORWARD_DAILY_EVENT_V1",
            "event_date": str(date.date()),
            "data_cutoff": _time(date, signal_cutoff_time, timezone),
            "signal": signal,
            "executions": day_executions,
            "daily_costs": {
                "commission": float(row["day_commission"]),
                "slippage_cost": float(row["day_slippage"]),
                "raw_notional": float(row["day_raw_notional"]),
            },
            "closed_cycles": closed_cycles,
            "closed_cycle_count": len(closed_cycles),
            "shadow_state": {
                "mode": str(row["mode"]),
                "tier": int(row["tier"]),
                "shares": int(row["shares"]),
            },
            "performance_metrics_disclosed": False,
            "previous_event_hash": previous,
        }
        record["event_hash"] = event_hash(record)
        previous = record["event_hash"]
        records.append(record)
    verify_event_chain(records)
    return records


def reconcile_append_only_events(
    existing: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """既有前缀必须逐字等价，只允许向后追加。"""

    verify_event_chain(existing)
    verify_event_chain(generated)
    if len(existing) > len(generated):
        raise ForwardLedgerError("生成账本短于既有账本，禁止截断")
    for index, prior in enumerate(existing):
        if prior != generated[index]:
            raise ForwardLedgerError(
                f"既有事件被改写：{prior.get('event_date', index)}"
            )
    return generated, len(generated) - len(existing)


__all__ = [
    "ForwardLedgerError",
    "build_daily_event_records",
    "event_hash",
    "reconcile_append_only_events",
    "verify_event_chain",
]

