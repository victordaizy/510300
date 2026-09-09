"""运行15币种小时横截面反转V8的冻结可见期。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.digital_asset_cross_sectional_hourly_reversal_v8 import (
    CONFIG,
    atomic_json,
    atomic_text,
    load_contract,
    run_visible,
)
from scripts.freeze_digital_asset_cross_sectional_hourly_reversal_v8 import verify


def write_failure_report(
    contract: dict[str, Any],
    verification: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    report = {
        "schema_version": "1.0.0",
        "report_id": "DIGITAL_ASSET_CROSS_SECTIONAL_HOURLY_REVERSAL_V8_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": "FAILED_DATA_OR_EXECUTION_CONTRACT",
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "manifest_verification": verification,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "performance_metrics_available": False,
        "decision": {
            "formula_family_replication_conditionally_eligible": False,
            "sealed_replication_authorized": False,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": "保留V8失败并关闭同候选复验；不得修改时域、选币数、成本、资金费率、容量或保证金参数救回",
        },
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(
        ROOT / outputs["visible_report_markdown"],
        "\n".join(
            [
                "# 多币种小时横截面反转V8可见期失败",
                "",
                "状态：`FAILED_DATA_OR_EXECUTION_CONTRACT`",
                "",
                f"错误类型：`{type(exc).__name__}`",
                "",
                f"错误：{exc}",
                "",
                "没有形成绩效；封存复验、Paper、Shadow、订单、账户连接和实盘均关闭。",
                "",
            ]
        ),
    )
    return report


def main() -> int:
    contract = load_contract(CONFIG)
    report_path = ROOT / contract["outputs"]["visible_report_json"]
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False))
        return 2 if str(existing.get("status", "")).startswith("FAILED") else 0
    verification = verify()
    if verification["failure_count"]:
        report = write_failure_report(
            contract,
            verification,
            RuntimeError(f"冻结清单验证失败：{verification['failures']}"),
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    try:
        report = run_visible(
            contract,
            manifest_verification=verification,
            bootstrap_repetitions_override=None,
        )
    except Exception as exc:
        report = write_failure_report(contract, verification, exc)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
