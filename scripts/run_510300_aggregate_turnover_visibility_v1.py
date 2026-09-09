"""运行冻结后的510300全市场换手率可见度V1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from aggregate_turnover_visibility_v1 import ContractError, run_study  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300全市场换手率可见度V1")
    parser.add_argument("--no-write", action="store_true", help="只复算，不写入产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_study(write=not arguments.no_write)
    except (ContractError, FileNotFoundError, ValueError, FileExistsError) as exc:
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
    except Exception as exc:  # pragma: no cover - 最终防线，保留PROGRAM_FAILED语义
        payload = {
            "status": "PROGRAM_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "mechanism_gate_passed": report["mechanism_gate_passed"],
        "portfolio_evaluated": report["portfolio_evaluated"],
        "return_evaluation": report["adjudication"]["return_evaluation"],
        "net_sharpe": report["adjudication"]["net_sharpe"],
        "historical_target_achieved": report["adjudication"][
            "historical_target_achieved"
        ],
        "verified_forward_target_achieved": False,
        "goal_achieved": False,
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

