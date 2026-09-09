"""运行冻结后的510300状态已知条件策略Oracle V1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from known_state_policy_oracle_v1 import ContractError, run_study  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300状态已知条件策略Oracle V1")
    parser.add_argument("--no-write", action="store_true", help="只复算，不写入产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_study(write=not arguments.no_write)
    except (ContractError, FileNotFoundError, ValueError, FileExistsError) as exc:
        payload = {
            "status": "NO_VIEW_ORACLE_DATA_OR_PROTOCOL_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "conditional_hypothesis": "NOT_EVALUATED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 保留PROGRAM_FAILED状态
        payload = {
            "status": "PROGRAM_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "conditional_hypothesis": "NOT_EVALUATED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "conditional_policy_hypothesis_supported": report["adjudication"][
            "conditional_policy_hypothesis_supported"
        ],
        "primary_base_net_sharpe": report["adjudication"][
            "primary_base_net_sharpe"
        ],
        "primary_stress_net_sharpe": report["adjudication"][
            "primary_stress_net_sharpe"
        ],
        "state_prediction_hypothesis": "NOT_EVALUATED",
        "historical_tradable_strategy_target_achieved": False,
        "goal_achieved": False,
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
