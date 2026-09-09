"""运行 510300 大盘状态识别与机制路由 V1 的冻结第一阶段研究。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from regime_transition_router_v1 import load_config, run_study  # noqa: E402


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    try:
        report = run_study(write=True)
    except Exception as error:  # noqa: BLE001 - 顶层运行器必须保留程序失败语义
        config = load_config()
        failure = {
            "project_id": config["protocol"]["project_id"],
            "status": "PROGRAM_FAILED_REGIME_TRANSITION_ROUTER_V1",
            "failed_at_asia_shanghai": datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "return_evaluation": "NOT_AVAILABLE_PROGRAM_FAILED",
            "strategy_backtest": "NOT_RUN",
            "phase_2_status": "SKIPPED_PROGRAM_FAILED",
            "live_trading_authorized": False,
        }
        failure_path = ROOT / config["paths"]["program_failure_json"]
        _atomic_json(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
