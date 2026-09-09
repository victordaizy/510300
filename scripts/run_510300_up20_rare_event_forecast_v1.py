"""运行冻结后的510300 UP20稀有事件预测V1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from up20_rare_event_forecast_v1 import ContractError, run_study  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300 UP20稀有事件预测V1")
    parser.add_argument("--no-write", action="store_true", help="完整复算但不写入产物")
    parser.add_argument("--quiet", action="store_true", help="不显示预序列拟合进度")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_study(
            write=not arguments.no_write,
            progress=not arguments.quiet,
        )
    except (ContractError, FileNotFoundError, FileExistsError, ValueError) as exc:
        payload = {
            "status": "NO_VIEW_UP20_FORECAST_DATA_OR_PROTOCOL_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "historical_account_simulation": "NOT_EVALUATED",
            "goal_achieved": False,
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 保留程序失败的精确状态
        payload = {
            "status": "PROGRAM_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "historical_account_simulation": "NOT_EVALUATED",
            "goal_achieved": False,
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    primary = report["primary_result"]
    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "primary_base_net_sharpe": primary["base"]["sharpe_zero_cash_rate"],
        "primary_stress_net_sharpe": primary["stress"]["sharpe_zero_cash_rate"],
        "recent_stress_net_sharpe": primary["recent_period"]["stress"]["net_sharpe"],
        "captured_up_blocks": primary["classification"]["captured_up_blocks"],
        "false_down_blocks": primary["classification"]["false_down_blocks"],
        "false_range_blocks": primary["classification"]["false_range_blocks"],
        "historical_account_simulation_target_achieved": report["adjudication"][
            "historical_account_simulation_target_achieved"
        ],
        "verified_forward_observations": 0,
        "goal_achieved": False,
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

