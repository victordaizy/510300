"""运行广度预热数据修复后的V1_0_1一次性预检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from macd_breadth_downside_preflight_v1_0_1 import (  # noqa: E402
    ContractError,
    run_preflight,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300机制预检V1_0_1")
    parser.add_argument("--no-write", action="store_true", help="复算但不写产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_preflight(write=not arguments.no_write)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        payload = {
            "status": "NO_VIEW_DATA_OR_PROTOCOL_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "passed": report["passed"],
        "return_evaluation": report["adjudication"]["return_evaluation"],
        "net_sharpe": report["adjudication"]["net_sharpe"],
        "target_net_sharpe": report["adjudication"]["target_net_sharpe"],
        "target_achieved": report["adjudication"]["target_achieved"],
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
