"""运行已冻结的510300 MACD×广度×下行波动一次性预检。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from macd_breadth_downside_preflight_v1 import ContractError, run_preflight  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行510300 MACD负柱收敛×官方权重广度×下行波动预检"
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="只在内存中复算，不写结果文件；仍会校验冻结清单",
    )
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
