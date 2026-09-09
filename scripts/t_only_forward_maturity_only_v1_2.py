"""T-only V1.1 成熟前公开输出收口。

本模块不改变冻结研究计算、参数、行情输入或影子账本。它只把人工可见的
状态与每日操作卡转换为成熟度/完整性口径，避免在达到 252 个新交易日且
闭合 3 个周期前披露收益、信号、仓位或门槛结果。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.json"
REPORT_MARKDOWN = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.md"
GUIDE_JSON = ROOT / "reports" / "forward" / "t_only_forward_v1_1_daily_guide.json"
GUIDE_MARKDOWN = ROOT / "reports" / "forward" / "t_only_forward_v1_1_daily_guide.md"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"
LEDGER = ROOT / "paper" / "t_only_forward_v1_1" / "daily_ledger.parquet"
LEGACY_RUN_STATUS = ROOT / "paper" / "t_only_forward_v1" / "daily_run_status.json"

SCHEMA_VERSION = "1.2.0"
ARTIFACT_ROLE = "T_ONLY_MATURITY_AND_COMPLETENESS_ONLY"
TRADING_DAYS_REQUIRED = 252
CLOSED_CYCLES_REQUIRED = 3

FORBIDDEN_PUBLIC_KEYS = {
    "metrics",
    "gates",
    "current_shadow_signal",
    "model_action",
    "shadow_account",
    "latest_indicator_diagnostic",
    "possible_model_actions",
    "daily_decision",
    "daily_headline",
    "decision",
    "headline",
    "weekly_state",
    "target_exposure",
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe",
    "maximum_drawdown",
    "average_exposure",
    "equity_cny",
    "shares",
    "tier",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON 顶层必须是对象：{path}")
    return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _normalise_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"完整性布尔字段类型非法：{value!r}")


def audit_ledger(ledger_path: Path, expected_rows: int) -> dict[str, Any]:
    if expected_rows < 0:
        raise RuntimeError("新交易日数量不得为负数")
    if not ledger_path.is_file():
        return {
            "expected_rows": expected_rows,
            "actual_rows": 0,
            "duplicate_dates": 0,
            "status": "PASS" if expected_rows == 0 else "FAILED_LEDGER_MISSING",
            "complete": expected_rows == 0,
            "sha256": None,
        }

    ledger = pd.read_parquet(ledger_path)
    if "date" not in ledger.columns:
        raise RuntimeError("T-only 影子账本缺少 date 列")
    dates = pd.to_datetime(ledger["date"], errors="raise").dt.normalize()
    actual_rows = int(len(ledger))
    duplicate_dates = int(dates.duplicated().sum())
    complete = actual_rows == expected_rows and duplicate_dates == 0
    return {
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "duplicate_dates": duplicate_dates,
        "status": "PASS" if complete else "FAILED_LEDGER_RECONCILIATION",
        "complete": complete,
        "sha256": sha256_file(ledger_path),
    }


def audit_data_gate(
    source_report: Mapping[str, Any],
    data_gate: Mapping[str, Any],
    *,
    data_gate_sha256: str | None = None,
) -> dict[str, Any]:
    report_market_date = source_report.get("as_of_market_date")
    gate_market_date = data_gate.get("actual_last_date")
    date_matches = bool(report_market_date and report_market_date == gate_market_date)
    frozen_input_unchanged = _normalise_bool(data_gate.get("frozen_input_unchanged"))
    dividend_check = data_gate.get("dividend_secondary_cross_check")
    dividend_status = (
        dividend_check.get("status") if isinstance(dividend_check, Mapping) else None
    )
    complete = bool(
        data_gate.get("status") == "PASS"
        and date_matches
        and frozen_input_unchanged is True
        and dividend_status == "PASS"
    )
    return {
        "status": "PASS" if complete else "FAILED_DATA_COMPLETENESS",
        "complete": complete,
        "data_gate_status": data_gate.get("status"),
        "as_of_market_date_matches": date_matches,
        "frozen_input_unchanged": frozen_input_unchanged,
        "dividend_cross_check_status": dividend_status,
        "data_gate_sha256": data_gate_sha256,
    }


def _source_counts(source_report: Mapping[str, Any]) -> tuple[int, int, dict[str, Any]]:
    new_days = int(source_report.get("new_trading_days", 0))
    closed_cycles = int(source_report.get("closed_cycles", 0))
    if new_days < 0 or closed_cycles < 0:
        raise RuntimeError("成熟度计数不得为负数")
    source_maturity = source_report.get("maturity")
    if not isinstance(source_maturity, Mapping):
        raise RuntimeError("T-only 状态缺少 maturity 对象")
    trading_days_required = int(
        source_maturity.get("trading_days_required", TRADING_DAYS_REQUIRED)
    )
    closed_cycles_required = int(
        source_maturity.get("closed_cycles_required", CLOSED_CYCLES_REQUIRED)
    )
    if trading_days_required != TRADING_DAYS_REQUIRED:
        raise RuntimeError("T-only 新交易日成熟门槛发生变化")
    if closed_cycles_required != CLOSED_CYCLES_REQUIRED:
        raise RuntimeError("T-only 闭合周期成熟门槛发生变化")
    computed_mature = bool(
        new_days >= TRADING_DAYS_REQUIRED
        and closed_cycles >= CLOSED_CYCLES_REQUIRED
    )
    declared_mature = bool(source_maturity.get("mature", False))
    if declared_mature != computed_mature:
        raise RuntimeError("T-only 成熟状态与冻结门槛计数不一致")
    return (
        new_days,
        closed_cycles,
        {
            "trading_days_required": TRADING_DAYS_REQUIRED,
            "closed_cycles_required": CLOSED_CYCLES_REQUIRED,
            "mature": computed_mature,
        },
    )


def build_maturity_only_snapshot(
    *,
    source_report: Mapping[str, Any],
    data_gate: Mapping[str, Any],
    ledger_audit: Mapping[str, Any],
    run_completeness: Mapping[str, Any] | None = None,
    source_detailed_status_sha256: str | None = None,
    data_gate_sha256: str | None = None,
) -> dict[str, Any]:
    """构造不含中途绩效或信号的权威公开状态。"""

    new_days, closed_cycles, maturity = _source_counts(source_report)
    data_audit = audit_data_gate(
        source_report, data_gate, data_gate_sha256=data_gate_sha256
    )
    ledger_complete = bool(ledger_audit.get("complete"))
    if run_completeness is None:
        run_audit: dict[str, Any] = {
            "status": "PENDING_DAILY_RUN_AUDIT",
            "complete": None,
            "run_id": None,
        }
    else:
        run_audit = dict(run_completeness)
        run_audit["complete"] = _normalise_bool(run_audit.get("complete"))

    if maturity["mature"]:
        status = "MATURE_REVIEW_REQUIRED"
        current_view = "MATURE_REVIEW_REQUIRED"
        evaluation_status = "MATURE_THRESHOLD_REACHED_REVIEW_SEPARATELY"
    elif new_days == 0:
        status = "COLLECTING_FORWARD_NOT_STARTED"
        current_view = "NO_VIEW_NO_FORWARD_TRADING_DAY"
        evaluation_status = "NOT_EVALUATED_BEFORE_MATURITY"
    else:
        status = "COLLECTING"
        current_view = "NO_VIEW_UNTIL_FORWARD_MATURITY"
        evaluation_status = "NOT_EVALUATED_BEFORE_MATURITY"

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "project_id": str(source_report.get("project_id") or "510300_T_ONLY_FORWARD_V1_1"),
        "source_protocol_version": "1.1.0",
        "operational_output_version": SCHEMA_VERSION,
        "status": status,
        "current_view": current_view,
        "as_of_market_date": source_report.get("as_of_market_date"),
        "forward_signal_start": source_report.get("forward_signal_start"),
        "new_trading_days": new_days,
        "closed_cycles": closed_cycles,
        "maturity": maturity,
        "completeness": {
            "data": data_audit,
            "ledger": dict(ledger_audit),
            "run": run_audit,
            "overall_complete": bool(
                data_audit["complete"]
                and ledger_complete
                and run_audit.get("complete") is True
            ),
        },
        "evaluation_status": evaluation_status,
        "source_detailed_status_sha256": source_detailed_status_sha256,
    }
    assert_maturity_only_payload(snapshot)
    return snapshot


def build_operational_failure_snapshot(
    *,
    previous_snapshot: Mapping[str, Any] | None,
    run_id: str,
    failure_class: str,
    failure_stage: str,
    failure_message: str,
) -> dict[str, Any]:
    previous = dict(previous_snapshot or {})
    maturity = previous.get("maturity")
    if not isinstance(maturity, Mapping):
        maturity = {
            "trading_days_required": TRADING_DAYS_REQUIRED,
            "closed_cycles_required": CLOSED_CYCLES_REQUIRED,
            "mature": False,
        }
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "project_id": str(previous.get("project_id") or "510300_T_ONLY_FORWARD_V1_1"),
        "source_protocol_version": "1.1.0",
        "operational_output_version": SCHEMA_VERSION,
        "status": "FAILED",
        "current_view": "NO_VIEW_OPERATIONAL_FAILURE",
        "as_of_market_date": previous.get("as_of_market_date"),
        "forward_signal_start": previous.get("forward_signal_start", "2026-08-19"),
        "new_trading_days": int(previous.get("new_trading_days", 0)),
        "closed_cycles": int(previous.get("closed_cycles", 0)),
        "maturity": dict(maturity),
        "completeness": {
            "data": {"status": "NOT_CONFIRMED_CURRENT_RUN", "complete": False},
            "ledger": {"status": "NOT_CONFIRMED_CURRENT_RUN", "complete": False},
            "run": {"status": "FAILED", "complete": False, "run_id": run_id},
            "overall_complete": False,
        },
        "evaluation_status": "NOT_EVALUATED_OPERATIONAL_FAILURE",
        "failure": {
            "class": failure_class,
            "stage": failure_stage,
            "message": failure_message,
        },
        "source_detailed_status_sha256": previous.get(
            "source_detailed_status_sha256"
        ),
    }
    assert_maturity_only_payload(snapshot)
    return snapshot


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def assert_maturity_only_payload(payload: Mapping[str, Any]) -> None:
    leaked = sorted(set(_walk_keys(payload)).intersection(FORBIDDEN_PUBLIC_KEYS))
    if leaked:
        raise RuntimeError(f"成熟度公开状态含有禁止字段：{', '.join(leaked)}")
    maturity = payload.get("maturity")
    if not isinstance(maturity, Mapping):
        raise RuntimeError("成熟度公开状态缺少 maturity")
    if not bool(maturity.get("mature")):
        allowed_views = {
            "NO_VIEW_NO_FORWARD_TRADING_DAY",
            "NO_VIEW_UNTIL_FORWARD_MATURITY",
            "NO_VIEW_OPERATIONAL_FAILURE",
        }
        if payload.get("current_view") not in allowed_views:
            raise RuntimeError("未成熟公开状态不是 NO_VIEW")


def public_guide(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    guide = {
        "schema_version": snapshot["schema_version"],
        "artifact_role": "T_ONLY_MATURITY_ONLY_DAILY_DELIVERABLE",
        "project_id": snapshot["project_id"],
        "status": snapshot["status"],
        "current_view": snapshot["current_view"],
        "as_of_market_date": snapshot["as_of_market_date"],
        "new_trading_days": snapshot["new_trading_days"],
        "closed_cycles": snapshot["closed_cycles"],
        "maturity": snapshot["maturity"],
        "completeness": snapshot["completeness"],
        "evaluation_status": snapshot["evaluation_status"],
    }
    if "failure" in snapshot:
        guide["failure"] = snapshot["failure"]
    assert_maturity_only_payload(guide)
    return guide


def render_markdown(snapshot: Mapping[str, Any]) -> str:
    completeness = snapshot["completeness"]
    data = completeness["data"]
    ledger = completeness["ledger"]
    run = completeness["run"]
    lines = [
        "# T_ONLY_FORWARD_V1.1 成熟度与完整性状态",
        "",
        f"- 状态：`{snapshot['status']}`",
        f"- 当前视图：`{snapshot['current_view']}`",
        f"- 行情截止：{snapshot['as_of_market_date']}",
        f"- 新交易日：{snapshot['new_trading_days']} / {snapshot['maturity']['trading_days_required']}",
        f"- 闭合周期：{snapshot['closed_cycles']} / {snapshot['maturity']['closed_cycles_required']}",
        f"- 数据完整性：`{data.get('status')}`",
        f"- 账本完整性：`{ledger.get('status')}`",
        f"- 本次运行完整性：`{run.get('status')}`",
        f"- 总体完整：`{str(bool(completeness['overall_complete'])).upper()}`",
        f"- 评价状态：`{snapshot['evaluation_status']}`",
        "",
        "达到 252 个新交易日且闭合 3 个周期前，本文件不提供收益、信号、仓位或临时优劣判断。",
        "",
    ]
    if "failure" in snapshot:
        failure = snapshot["failure"]
        lines.extend(
            [
                "## 本次失败",
                "",
                f"- 分类：`{failure['class']}`",
                f"- 阶段：`{failure['stage']}`",
                f"- 原因：{failure['message']}",
                "",
            ]
        )
    return "\n".join(lines)


def publish_snapshot(
    snapshot: Mapping[str, Any],
    *,
    report_json: Path = REPORT_JSON,
    report_markdown: Path = REPORT_MARKDOWN,
    guide_json: Path = GUIDE_JSON,
    guide_markdown: Path = GUIDE_MARKDOWN,
) -> None:
    assert_maturity_only_payload(snapshot)
    guide = public_guide(snapshot)
    markdown = render_markdown(snapshot)
    atomic_json(report_json, snapshot)
    atomic_text(report_markdown, markdown)
    atomic_json(guide_json, guide)
    atomic_text(guide_markdown, markdown)


def retire_legacy_run_status(
    snapshot: Mapping[str, Any], *, legacy_run_status: Path = LEGACY_RUN_STATUS
) -> dict[str, Any]:
    """把旧日更状态替换为无信号、无操作标题的迁移指针。"""

    previous_sha256 = (
        sha256_file(legacy_run_status) if legacy_run_status.is_file() else None
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "T_ONLY_LEGACY_RUN_STATUS_RETIRED",
        "status": "SUPERSEDED_BY_MATURITY_ONLY_V1_2",
        "current_view": snapshot["current_view"],
        "as_of_market_date": snapshot["as_of_market_date"],
        "new_trading_days": snapshot["new_trading_days"],
        "closed_cycles": snapshot["closed_cycles"],
        "maturity": snapshot["maturity"],
        "completeness": snapshot["completeness"],
        "evaluation_status": snapshot["evaluation_status"],
        "authoritative_status": REPORT_JSON.relative_to(ROOT).as_posix(),
        "source_legacy_run_status_sha256": previous_sha256,
    }
    assert_maturity_only_payload(payload)
    atomic_json(legacy_run_status, payload)
    return payload


def sanitize_existing_outputs(
    *,
    run_completeness: Mapping[str, Any] | None = None,
    report_json: Path = REPORT_JSON,
    report_markdown: Path = REPORT_MARKDOWN,
    guide_json: Path = GUIDE_JSON,
    guide_markdown: Path = GUIDE_MARKDOWN,
    data_gate_path: Path = DATA_GATE,
    ledger_path: Path = LEDGER,
) -> dict[str, Any]:
    if not report_json.is_file():
        raise RuntimeError(f"T-only V1.1 状态缺失：{report_json}")
    if not data_gate_path.is_file():
        raise RuntimeError(f"T-only 数据闸门缺失：{data_gate_path}")
    source_report = read_json(report_json)
    data_gate = read_json(data_gate_path)

    if source_report.get("schema_version") == SCHEMA_VERSION:
        assert_maturity_only_payload(source_report)
        source_sha256 = source_report.get("source_detailed_status_sha256")
    else:
        source_sha256 = sha256_file(report_json)

    new_days, _, _ = _source_counts(source_report)
    ledger_result = audit_ledger(ledger_path, new_days)
    snapshot = build_maturity_only_snapshot(
        source_report=source_report,
        data_gate=data_gate,
        ledger_audit=ledger_result,
        run_completeness=run_completeness,
        source_detailed_status_sha256=(
            str(source_sha256) if source_sha256 is not None else None
        ),
        data_gate_sha256=sha256_file(data_gate_path),
    )
    publish_snapshot(
        snapshot,
        report_json=report_json,
        report_markdown=report_markdown,
        guide_json=guide_json,
        guide_markdown=guide_markdown,
    )
    return snapshot


__all__ = [
    "ARTIFACT_ROLE",
    "CLOSED_CYCLES_REQUIRED",
    "DATA_GATE",
    "FORBIDDEN_PUBLIC_KEYS",
    "GUIDE_JSON",
    "GUIDE_MARKDOWN",
    "LEDGER",
    "REPORT_JSON",
    "REPORT_MARKDOWN",
    "SCHEMA_VERSION",
    "TRADING_DAYS_REQUIRED",
    "assert_maturity_only_payload",
    "atomic_json",
    "build_maturity_only_snapshot",
    "build_operational_failure_snapshot",
    "publish_snapshot",
    "read_json",
    "retire_legacy_run_status",
    "sanitize_existing_outputs",
    "sha256_file",
]
