"""运行已冻结的510300压力传导与耗竭机制图谱V1。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from stress_transmission_and_exhaustion_atlas_v1 import (  # noqa: E402
    ContractError,
    load_config,
    project_path,
    run_study,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行510300压力传导与耗竭机制图谱V1")
    parser.add_argument("--no-write", action="store_true", help="复算但不写正式产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        report = run_study(write=not arguments.no_write)
    except (
        ContractError,
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        OSError,
        np.linalg.LinAlgError,
    ) as exc:
        payload = {
            "project_id": "510300_STRESS_TRANSMISSION_AND_EXHAUSTION_ATLAS_V1",
            "status": "PROGRAM_FAILED_STRESS_TRANSMISSION_AND_EXHAUSTION_ATLAS_V1",
            "generated_at_asia_shanghai": datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "strategy_backtest": "NOT_RUN",
            "position_mapping": "DISABLED",
            "current_signal": "NOT_CREATED",
            "live_trading_authorized": False,
        }
        if not arguments.no_write:
            try:
                config = load_config()
                _atomic_json(project_path(config["paths"]["program_failure_json"]), payload)
            except (ContractError, FileNotFoundError, KeyError, OSError, ValueError):
                pass
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    summary = {
        "project_id": report["project_id"],
        "status": report["status"],
        "data_contract_passed": report["data_admission"]["passed"],
        "router_events": report.get("events", {}).get("router_event_count"),
        "price_damage_events": report.get("events", {}).get("price_damage_event_count"),
        "all_four_gates_passed": report.get("gates", {}).get(
            "all_four_gates_passed", False
        ),
        "return_evaluation": report["return_evaluation"],
        "strategy_backtest": report["strategy_backtest"],
        "result_written": not arguments.no_write,
        "current_signal": "NOT_CREATED",
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
