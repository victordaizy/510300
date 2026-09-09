"""运行T_ONLY不可变每日事件账本；只展示完整性，不展示正式绩效。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.t_only_forward_event_ledger import (
    ForwardLedgerError,
    build_daily_event_records,
    reconcile_append_only_events,
)
from research.t_only_forward_v1_1 import (
    apply_causal_weekly_states,
    load_calendar,
    load_config as load_v1_1_config,
)
from research.t_only_robustness_forward_v1 import (
    load_stress_config,
    simulate_t_only,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config as load_base_config,
)
from scripts.run_t_only_forward_v1 import load_dividends
from scripts.run_t_only_forward_v1_1 import verify_manifest as verify_v1_1_manifest


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_FILE = ROOT / "config" / "t_only_forward_integrity_v1.yaml"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ForwardLedgerError(f"事件账本第{line_number}行不是JSON对象")
        records.append(value)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_text(path, content)


def _status_markdown(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# T_ONLY 前瞻事件账本完整性状态",
            "",
            f"- 状态：`{status['status']}`",
            f"- 行情截止：{status['as_of_market_date']}",
            f"- 事件账本天数：{status['event_count']}",
            f"- 闭合周期：{status['closed_cycle_count']}",
            f"- 哈希链：`{status['chain_integrity']}`",
            f"- 链头：`{status['chain_head'] or '—'}`",
            "- 252日/3周期只表示运行成熟度，不是收益通过门。",
            "- 正式评价固定为最多1,500个交易日、至少20个闭合周期，并且只评价一次。",
            "- 运行团队只读取完整性、信号时点、执行/成本和周期计数；正式绩效保持盲化。",
            "- 真实仓位映射、订单、券商连接和实盘全部关闭。",
            "",
        ]
    )


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    verify_v1_1_manifest()
    start_audit = ROOT / config["frozen_start_audit"]["path"]
    if _sha256(start_audit) != config["frozen_start_audit"]["sha256"]:
        raise RuntimeError("NO_VIEW_T_ONLY_START_AUDIT_HASH_MISMATCH")

    v1_1_config = load_v1_1_config()
    parent_config = load_stress_config()
    base_config = load_base_config()
    forward = parent_config["forward_data"]
    market_path = ROOT / forward["refreshed_market_file"]
    metadata_path = ROOT / forward["metadata_file"]
    if not DATA_GATE.exists() or not market_path.exists() or not metadata_path.exists():
        raise RuntimeError("NO_VIEW_DATA_NOT_REFRESHED")
    data_gate = json.loads(DATA_GATE.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS" or metadata.get("status") != "PASS":
        raise RuntimeError("NO_VIEW_DATA_GATE_FAILED")
    if _sha256(market_path) != metadata.get("sha256"):
        raise RuntimeError("NO_VIEW_REFRESHED_MARKET_HASH_MISMATCH")

    market = pd.read_parquet(market_path)
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    dividends = load_dividends(ROOT / forward["dividend_file"])
    features, weekly = build_features(market, dividends, base_config)
    calendar = load_calendar(v1_1_config)
    features, causality_audit = apply_causal_weekly_states(
        features,
        weekly,
        calendar,
        signal_start=calendar["trade_date"].min(),
    )
    signal_start = pd.Timestamp(config["forward_signal_start"])
    forward_features = features.loc[features["date"].ge(signal_start)].reset_index(
        drop=True
    )
    action_targets = {
        reason: float(value["target_exposure"])
        for reason, value in v1_1_config["daily_guide"]["model_actions"].items()
    }
    if forward_features.empty:
        generated: list[dict[str, Any]] = []
    else:
        simulation = simulate_t_only(
            forward_features,
            dividends,
            base_config,
            scenario_id="T_ONLY_FORWARD_INTEGRITY_V1",
            allow_terminal_pending_signal=True,
        )
        generated = build_daily_event_records(
            daily_ledger=simulation["ledger"],
            executions=simulation["executions"],
            cycles=simulation["cycles"],
            calendar=calendar,
            action_targets=action_targets,
            timezone=config["timezone"],
            signal_cutoff_time=config["signal_cutoff_time"],
            signal_record_time=config["signal_record_time"],
            next_open_time=config["next_open_time"],
        )

    ledger_path = ROOT / config["outputs"]["event_ledger"]
    existing = _read_jsonl(ledger_path)
    records, appended = reconcile_append_only_events(existing, generated)
    if appended > 1:
        raise RuntimeError("FAILED_FORWARD_LEDGER_GAP：单次运行不得事后补写多个交易日")
    if appended or not ledger_path.exists():
        _write_jsonl(ledger_path, records)

    event_count = len(records)
    closed_cycle_count = sum(int(item["closed_cycle_count"]) for item in records)
    operational = config["operational_maturity"]
    formal = config["formal_power_plan"]
    operational_mature = bool(
        event_count >= int(operational["minimum_new_trading_days"])
        and closed_cycle_count >= int(operational["minimum_closed_cycles"])
    )
    formal_eligible = bool(
        event_count >= int(formal["maximum_forward_trading_days"])
        and closed_cycle_count >= int(formal["minimum_closed_cycles"])
    )
    status = {
        "project_id": config["version"],
        "parent_project_id": config["parent_project_id"],
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "status": "COLLECTING" if event_count else "COLLECTING_FORWARD_NOT_STARTED",
        "current_view": "NO_VIEW_UNTIL_FIXED_FORMAL_EVALUATION",
        "as_of_market_date": metadata["actual_last_date"],
        "event_count": event_count,
        "closed_cycle_count": closed_cycle_count,
        "events_appended_this_run": appended,
        "chain_integrity": "PASS",
        "chain_head": records[-1]["event_hash"] if records else None,
        "ledger_continuity": "PASS",
        "weekly_causality_audit": causality_audit,
        "operational_maturity": {
            **operational,
            "mature": operational_mature,
            "is_performance_pass_gate": False,
        },
        "formal_power_plan": {
            **formal,
            "eligible": formal_eligible,
            "evaluated": False,
        },
        "performance_visibility": "BLINDED_UNTIL_FIXED_FORMAL_EVALUATION",
        "performance_metrics": None,
        "safety": config["governance"],
    }
    status_json = ROOT / config["outputs"]["status_json"]
    status_markdown = ROOT / config["outputs"]["status_markdown"]
    _atomic_text(status_json, json.dumps(status, ensure_ascii=False, indent=2))
    _atomic_text(status_markdown, _status_markdown(status))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

