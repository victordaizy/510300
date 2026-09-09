"""运行中国可转债传统双低周频轮动V1冻结可见期。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cb_double_low_rotation_v1 import (
    CONFIG,
    atomic_json,
    atomic_text,
    load_contract,
    run_visible,
)
from scripts.freeze_cb_double_low_rotation_v1 import verify


def write_failure_report(
    contract: dict[str, Any],
    manifest_verification: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    """保留数据或执行失败，不伪造绩效，也不打开封存复验。"""

    report = {
        "schema_version": "1.0.0",
        "report_id": "CB_DOUBLE_LOW_ROTATION_V1_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": "FAILED_DATA_OR_EXECUTION_CONTRACT",
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "manifest_verification": manifest_verification,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "performance_metrics_available": False,
        "decision": {
            "formula_family_replication_conditionally_eligible": False,
            "sealed_replication_authorized": False,
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": "保留失败并关闭复验；若是实现合同错误，只能另起版本修正",
        },
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(
        ROOT / outputs["visible_report_markdown"],
        "\n".join(
            [
                "# 中国可转债传统双低周频轮动V1可见期失败",
                "",
                "状态：`FAILED_DATA_OR_EXECUTION_CONTRACT`",
                "",
                f"错误类型：`{type(exc).__name__}`",
                "",
                f"错误：{exc}",
                "",
                "没有形成绩效；封存复验、Paper、Shadow、订单和实盘均关闭。",
                "",
            ]
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="运行可转债双低周频轮动V1可见期")
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=None,
        help="仅供合成测试；正式运行省略以使用冻结的5000次",
    )
    args = parser.parse_args()
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
            bootstrap_repetitions_override=args.bootstrap_repetitions,
        )
    except Exception as exc:
        report = write_failure_report(contract, verification, exc)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
