"""T_ONLY_FORWARD_V1.1：交易日历约束的因果周线与每日操作卡。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "t_only_forward_v1_1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("V1.1配置必须是YAML对象")
    protocol = config["protocol"]
    if protocol["historical_strategy_parameters_changed"] is not False:
        raise ValueError("V1.1不得修改历史策略参数")
    if protocol["forward_signal_start"] != "2026-08-19":
        raise ValueError("V1.1前瞻起点不得变化")
    governance = config["governance"]
    if any(
        governance[key]
        for key in (
            "live_position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
        )
    ):
        raise ValueError("V1.1安全开关必须保持关闭")
    return config


def load_calendar(config: dict[str, Any]) -> pd.DataFrame:
    contract = config["calendar_contract"]
    calendar_path = ROOT / contract["file"]
    metadata_path = ROOT / contract["metadata_file"]
    if not calendar_path.exists() or not metadata_path.exists():
        raise RuntimeError("NO_VIEW_CALENDAR_INPUT_MISSING")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS":
        raise RuntimeError("NO_VIEW_CALENDAR_GATE_FAILED")
    if metadata.get("official_source") != contract["required_source"]:
        raise RuntimeError("NO_VIEW_CALENDAR_SOURCE_MISMATCH")
    if int(metadata.get("year")) != int(contract["required_year"]):
        raise RuntimeError("NO_VIEW_CALENDAR_YEAR_MISMATCH")
    if sha256(calendar_path) != metadata.get("sha256"):
        raise RuntimeError("NO_VIEW_CALENDAR_HASH_MISMATCH")
    calendar = pd.read_csv(calendar_path)
    calendar["trade_date"] = pd.to_datetime(
        calendar["trade_date"], errors="raise"
    ).dt.normalize()
    if calendar["trade_date"].duplicated().any():
        raise RuntimeError("NO_VIEW_CALENDAR_DUPLICATE_DATE")
    if not calendar["trade_date"].is_monotonic_increasing:
        raise RuntimeError("NO_VIEW_CALENDAR_NOT_SORTED")
    return calendar


def apply_causal_weekly_states(
    features: pd.DataFrame,
    weekly: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    signal_start: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """仅在官方交易周最后交易日启用本周周线状态。"""

    corrected = features.copy()
    corrected["date"] = pd.to_datetime(corrected["date"]).dt.normalize()
    calendar_dates = set(calendar["trade_date"].tolist())
    forward_dates = corrected.loc[
        corrected["date"] >= signal_start, "date"
    ].tolist()
    missing_dates = [date for date in forward_dates if date not in calendar_dates]
    if missing_dates:
        raise RuntimeError(
            "NO_VIEW_MARKET_DATE_NOT_IN_FROZEN_CALENDAR："
            + ",".join(str(date.date()) for date in missing_dates)
        )
    if forward_dates and calendar["trade_date"].max() < max(forward_dates):
        raise RuntimeError("NO_VIEW_CALENDAR_COVERAGE_STALE")

    corrected["v1_uncorrected_weekly_state_at_close"] = corrected[
        "weekly_state_at_close"
    ]
    corrected["v1_uncorrected_weekly_state_source_date"] = corrected[
        "weekly_state_source_date"
    ]
    corrected["v1_uncorrected_is_completed_week_close"] = corrected[
        "is_completed_week_close"
    ]

    current_state = dict(zip(weekly["week_period"], weekly["state"]))
    previous_state = dict(
        zip(weekly["week_period"], weekly["state"].shift(1).fillna("W_UNKNOWN"))
    )
    previous_source = dict(
        zip(weekly["week_period"], weekly["last_date"].shift(1))
    )
    official_last = dict(
        zip(
            calendar["trade_date"].dt.to_period("W-FRI"),
            calendar.groupby(calendar["trade_date"].dt.to_period("W-FRI"))[
                "trade_date"
            ].transform("max"),
        )
    )

    audit_rows = 0
    changed_rows = 0
    for index, row in corrected.loc[corrected["date"] >= signal_start].iterrows():
        period = row["date"].to_period("W-FRI")
        if period not in official_last:
            raise RuntimeError(f"NO_VIEW_WEEK_NOT_IN_FROZEN_CALENDAR：{period}")
        is_complete = row["date"] == official_last[period]
        state = current_state[period] if is_complete else previous_state[period]
        source = row["date"] if is_complete else previous_source[period]
        if str(row["weekly_state_at_close"]) != str(state):
            changed_rows += 1
        corrected.at[index, "weekly_state_at_close"] = state
        corrected.at[index, "weekly_state_source_date"] = source
        corrected.at[index, "is_completed_week_close"] = is_complete
        audit_rows += 1
    audit = {
        "status": "PASS",
        "forward_rows_checked": audit_rows,
        "rows_with_state_change_vs_v1_runtime": changed_rows,
        "calendar_coverage_end": str(calendar["trade_date"].max().date()),
        "rule": "ONLY_OFFICIAL_WEEK_LAST_TRADE_DATE_USES_CURRENT_WEEK_STATE",
    }
    return corrected, audit


def latest_diagnostic(features: pd.DataFrame) -> dict[str, Any]:
    row = features.iloc[-1]
    close = float(row["adjusted_close"])
    hh20 = float(row["hh20"]) if pd.notna(row["hh20"]) else None
    ll10 = float(row["ll10"]) if pd.notna(row["ll10"]) else None
    atr14 = float(row["atr14"]) if pd.notna(row["atr14"]) else None
    weekly_state = str(row["weekly_state_at_close"])
    entry_ready = bool(
        weekly_state == "W_BULL" and hh20 is not None and close > hh20
    )
    return {
        "date": str(pd.Timestamp(row["date"]).date()),
        "raw_close": float(row["close"]),
        "adjusted_close": close,
        "weekly_state_at_close": weekly_state,
        "weekly_state_source_date": (
            str(pd.Timestamp(row["weekly_state_source_date"]).date())
            if pd.notna(row["weekly_state_source_date"])
            else None
        ),
        "is_completed_week_close": bool(row["is_completed_week_close"]),
        "donchian_entry_hh20": hh20,
        "donchian_exit_ll10": ll10,
        "atr14": atr14,
        "trend_entry_condition_met": entry_ready,
        "close_to_entry_threshold_gap": (
            float(close / hh20 - 1.0) if hh20 is not None else None
        ),
    }


def build_daily_guide(
    *,
    status_report: dict[str, Any],
    diagnostic: dict[str, Any],
    config: dict[str, Any],
    shadow_state: dict[str, Any],
) -> dict[str, Any]:
    status = str(status_report["status"])
    signal = status_report.get("current_shadow_signal")
    if status == "COLLECTING_FORWARD_NOT_STARTED":
        decision = "NO_ACTION_FORWARD_NOT_STARTED"
        headline = "前瞻观察尚未开始，今天不产生模型动作。"
        action = None
    elif signal is None:
        decision = "NO_ACTION_NO_SIGNAL"
        headline = "数据有效，但今天收盘没有生成下一交易日影子动作。"
        action = None
    else:
        reason = str(signal["reason"])
        registered = config["daily_guide"]["model_actions"].get(reason)
        if registered is None:
            raise RuntimeError(f"NO_VIEW_UNREGISTERED_DAILY_ACTION：{reason}")
        decision = "SHADOW_ACTION_PENDING_NEXT_OPEN"
        headline = str(registered["label"])
        action = {
            "signal_date": signal["signal_date"],
            "reason": reason,
            "label": registered["label"],
            "target_exposure": float(registered["target_exposure"]),
            "execution_timing": "NEXT_TRADING_DAY_OPEN_SHADOW_ONLY",
            "share_quantity": None,
            "share_quantity_reason": "下一交易日开盘价未知且系统不读取真实持仓",
        }
    return {
        "project_id": config["protocol"]["project_id"],
        "as_of_market_date": status_report["as_of_market_date"],
        "forward_status": status,
        "current_view": status_report["current_view"],
        "decision": decision,
        "headline": headline,
        "model_action": action,
        "shadow_account": shadow_state,
        "latest_indicator_diagnostic": diagnostic,
        "daily_checklist": [
            "确认每日运行状态退出码为0且数据闸门为PASS",
            "确认行情截止日是最新完整交易日，禁止使用昨日状态冒充今日",
            "只读取current_shadow_signal，不凭盘中感觉提前猜测收盘信号",
            "信号在收盘形成，只能在下一交易日开盘进入影子执行",
            "正常调仓必须满足100份整数倍和单腿至少5000元；风险清仓可豁免5000元门槛",
            "股票ETF按T+1管理，当日买入份额不用于当日卖出",
            "NO_VIEW或运行失败时不沿用上一日操作卡",
        ],
        "possible_model_actions": config["daily_guide"]["model_actions"],
        "real_money_boundary": {
            "system_authorization": "NOT_AUTHORIZED",
            "reason": "前瞻样本未成熟，且系统未读取真实持仓、可用资金和可卖份额",
            "user_decision_required": True,
        },
        "safety": config["governance"],
    }


def guide_markdown(guide: dict[str, Any]) -> str:
    diagnostic = guide["latest_indicator_diagnostic"]
    action = guide["model_action"]
    lines = [
        "# 510300 T_ONLY_FORWARD_V1.1 每日操作卡",
        "",
        f"- 行情截止：{guide['as_of_market_date']}",
        f"- 系统状态：`{guide['forward_status']}`",
        f"- 今日结论：**{guide['headline']}**",
        f"- 使用边界：`{guide['real_money_boundary']['system_authorization']}`（影子验证）",
        "",
        "## 今天具体怎么看",
        "",
        f"- 周线状态：`{diagnostic['weekly_state_at_close']}`，来源于 {diagnostic['weekly_state_source_date']}。",
        f"- 当天是否为完整周最后交易日：{diagnostic['is_completed_week_close']}。",
        f"- 收盘价：{diagnostic['raw_close']:.3f}；前20日最高价：{diagnostic['donchian_entry_hh20']:.3f}。",
        f"- 趋势入场条件是否满足：{diagnostic['trend_entry_condition_met']}。",
        "",
        "## 下一交易日影子动作",
        "",
    ]
    if action is None:
        lines.append("无。不要依据盘中感觉自行补出信号。")
    else:
        lines.extend(
            [
                f"- 信号日：{action['signal_date']}",
                f"- 原因：`{action['reason']}`",
                f"- 动作：{action['label']}",
                f"- 模型目标仓位：{action['target_exposure']:.1%}",
                "- 数量：暂不生成；下一开盘价未知，且系统不读取真实账户。",
            ]
        )
    lines.extend(
        [
            "",
            "## 每天固定检查",
            "",
            *[f"- {item}" for item in guide["daily_checklist"]],
            "",
            "## 重要说明",
            "",
            "这是冻结策略的每日影子操作卡，不是券商订单。前瞻样本达到252个新交易日和3个闭合周期前，系统不宣称策略已经通过验证。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "apply_causal_weekly_states",
    "build_daily_guide",
    "guide_markdown",
    "latest_diagnostic",
    "load_calendar",
    "load_config",
    "sha256",
]
