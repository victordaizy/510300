"""运行冻结后的510300未来20日上涨机会信息质量预算V1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from oracle_information_budget_v1 import ContractError, run_study  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300上涨机会信息质量预算V1")
    parser.add_argument("--no-write", action="store_true", help="完整复算但不写入产物")
    parser.add_argument("--quiet", action="store_true", help="不显示阶段进度")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_study(
            write=not arguments.no_write,
            progress=not arguments.quiet,
        )
    except (ContractError, FileNotFoundError, ValueError, FileExistsError) as exc:
        payload = {
            "status": "NO_VIEW_INFORMATION_BUDGET_DATA_OR_PROTOCOL_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "information_budget": "NOT_EVALUATED",
            "realistic_up20_forecast": "NOT_EVALUATED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 保留程序失败的精确状态
        payload = {
            "status": "PROGRAM_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "information_budget": "NOT_EVALUATED",
            "realistic_up20_forecast": "NOT_EVALUATED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "information_budget": report["information_budget"],
        "conservative_up20_forecast_research_advancement_gate_passed": report[
            "adjudication"
        ]["conservative_up20_forecast_research_advancement_gate_passed"],
        "realistic_up20_forecast": "NOT_EVALUATED",
        "historical_tradable_strategy_target_achieved": False,
        "goal_achieved": False,
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
