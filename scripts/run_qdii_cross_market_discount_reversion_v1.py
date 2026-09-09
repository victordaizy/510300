"""运行QDII跨市场折价回归V1的冻结可见期。"""

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

from research.qdii_cross_market_discount_reversion_v1 import (
    CONFIG,
    atomic_json,
    atomic_text,
    load_contract,
    run_visible,
)
from scripts.freeze_qdii_cross_market_discount_reversion_v1 import verify


def write_failure_report(
    contract: dict[str, Any],
    manifest_verification: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    """保留失败，不输出伪造绩效。"""

    report = {
        "schema_version": "1.0.0",
        "report_id": "QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_VISIBLE",
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
            "sealed_replication_open": False,
            "candidate_may_be_reparameterized_after_failure": False,
            "next_step": "保留失败，不打开封存期，不调参救回",
        },
        "safety": contract["safety"],
    }
    json_path = ROOT / contract["outputs"]["visible_report_json"]
    markdown_path = ROOT / contract["outputs"]["visible_report_markdown"]
    atomic_json(json_path, report)
    atomic_text(
        markdown_path,
        "\n".join(
            [
                "# QDII跨市场折价回归V1可见期失败",
                "",
                "状态：`FAILED_DATA_OR_EXECUTION_CONTRACT`",
                "",
                f"错误类型：`{type(exc).__name__}`",
                "",
                f"错误：{exc}",
                "",
                "未计算绩效，封存期未打开，Paper、Shadow、订单和实盘均关闭。",
                "",
            ]
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="运行QDII跨市场折价回归V1可见期")
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=None,
        help="仅用于测试；正式运行省略以使用冻结的5000次",
    )
    args = parser.parse_args()
    contract = load_contract(CONFIG)
    verification = verify()
    if verification["failure_count"]:
        exc = RuntimeError(f"冻结清单验证失败：{verification['failures']}")
        report = write_failure_report(contract, verification, exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    try:
        report = run_visible(
            contract,
            manifest_verification=verification,
            bootstrap_repetitions_override=args.bootstrap_repetitions,
        )
    except Exception as exc:
        report = write_failure_report(contract, verification, exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
