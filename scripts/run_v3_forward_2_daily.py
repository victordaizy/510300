"""运行零Token的V3_FORWARD_2日终前瞻周期。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v3_forward_2_validation import CONFIG_FILE, run_forward_cycle
from scripts.refresh_v3_forward_2_inputs import refresh_all


RUN_STATUS_FILE = ROOT / "paper" / "v3_forward_2_daily_run_status.json"


def _persist_run_status(status: dict[str, Any]) -> None:
    RUN_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUN_STATUS_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(RUN_STATUS_FILE)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行V3_FORWARD_2零Token日终周期")
    parser.add_argument("--date", help="目标日期YYYY-MM-DD；禁止历史回填")
    parser.add_argument("--skip-refresh", action="store_true", help="仅供测试")
    parser.add_argument("--force-report", action="store_true", help="仅供受控审计")
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    signal_date = pd.Timestamp(args.date) if args.date else pd.Timestamp(now.date())
    if signal_date.normalize() != pd.Timestamp(now.date()):
        raise ValueError("禁止历史回填：目标日期必须等于上海本地日期")
    overall_started = perf_counter()
    status: dict[str, Any] = {
        "signal_date": str(signal_date.date()),
        "started_at": now.isoformat(),
        "status": "RUNNING",
        "active_step": None,
        "steps": {},
        "live_trading_authorized": False,
    }
    _persist_run_status(status)
    try:
        if not args.skip_refresh:
            step_started = perf_counter()
            status["active_step"] = "input_refresh"
            status["steps"]["input_refresh"] = {"status": "RUNNING"}
            _persist_run_status(status)
            refresh = refresh_all(signal_date)
            status["steps"]["input_refresh"] = {
                "status": refresh["status"],
                "duration_seconds": perf_counter() - step_started,
                "detail_status_file": "paper/v3_forward_2_input_refresh_status.json",
            }
            _persist_run_status(status)
            if refresh["status"] != "SUCCESS":
                status["status"] = refresh["status"]
                print(json.dumps(refresh, ensure_ascii=False, indent=2, default=str))
                return 0
        config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
        start = pd.Timestamp(config["model"]["true_oos_start"])
        if signal_date < start:
            result = {
                "model_version": config["model"]["model_version"],
                "cycle_status": "PRESTART_INPUTS_REFRESHED",
                "true_oos_start": str(start.date()),
                "message": "仅完成免费数据链预热；未生成信号。",
            }
            status["status"] = result["cycle_status"]
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        step_started = perf_counter()
        status["active_step"] = "ridge_and_forward_output"
        status["steps"]["ridge_and_forward_output"] = {"status": "RUNNING"}
        _persist_run_status(status)
        result = run_forward_cycle(
            signal_date=signal_date,
            generated_at=now,
            enforce_same_local_date=True,
            force_report=args.force_report,
        )
        status["steps"]["ridge_and_forward_output"] = {
            "status": "SUCCESS",
            "duration_seconds": perf_counter() - step_started,
            "cycle_status": result.get("cycle_status"),
        }
        status["status"] = "SUCCESS"
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        status["status"] = "FAILED"
        status["failed_step"] = status["active_step"]
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["duration_seconds"] = perf_counter() - overall_started
        status["active_step"] = None if status["status"] != "RUNNING" else status["active_step"]
        _persist_run_status(status)


if __name__ == "__main__":
    raise SystemExit(main())
