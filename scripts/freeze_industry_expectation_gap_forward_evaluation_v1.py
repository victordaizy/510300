"""冻结行业预期差 V1 的前瞻评价协议。"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.industry_expectation_gap_forward_evaluation import (  # noqa: E402
    ForwardEvaluationError,
    build_forward_evaluation_report,
    determine_observation_state,
    file_sha256,
    load_trading_calendar,
    resolve_maturity_schedule,
    validate_evaluation_config,
    verify_hash_manifest,
)


CONFIG_RELATIVE_PATH = "config/industry_expectation_gap_forward_evaluation_v1.yaml"
FROZEN_FILES = [
    "docs/INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_SPEC.md",
    CONFIG_RELATIVE_PATH,
    "research/industry_expectation_gap_forward_evaluation.py",
    "scripts/run_industry_expectation_gap_forward_evaluation_v1.py",
    "scripts/freeze_industry_expectation_gap_forward_evaluation_v1.py",
    "tests/test_industry_expectation_gap_forward_evaluation.py",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ForwardEvaluationError(f"YAML 顶层必须是对象：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ForwardEvaluationError(f"JSON 顶层必须是对象：{path}")
    return value


def _path(relative: str) -> Path:
    root = WORKSPACE_ROOT.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ForwardEvaluationError(f"路径越出工作区：{relative}") from exc
    return result


def _git_text(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _frame_record(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256_at_freeze": file_sha256(path),
        "rows_at_freeze": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "date" in frame.columns and len(frame):
        dates = pd.to_datetime(frame["date"], errors="coerce")
        record["date_min_at_freeze"] = dates.min().date().isoformat()
        record["date_max_at_freeze"] = dates.max().date().isoformat()
    return record


def main() -> int:
    config = _load_yaml(_path(CONFIG_RELATIVE_PATH))
    rules, safety = validate_evaluation_config(config)
    frozen = config["frozen_prediction"]
    verify_hash_manifest(
        WORKSPACE_ROOT,
        frozen["manifest"],
        frozen["manifest_sha256"],
    )
    frozen_result = _load_json(_path(frozen["result"]))

    input_config = config["mutable_outcome_inputs"]
    calendar_paths = sorted(WORKSPACE_ROOT.glob(input_config["trading_calendar_glob"]))
    if not calendar_paths:
        raise ForwardEvaluationError("冻结时没有交易日历")
    calendar_frames = [pd.read_csv(path) for path in calendar_paths]
    calendar = load_trading_calendar(calendar_frames)
    schedule = resolve_maturity_schedule(
        calendar,
        config["entry_date"],
        rules.horizons_trading_days,
        config.get("pre_resolved_maturity_dates"),
    )

    constituent_path = _path(input_config["constituent_total_return_daily"])
    weights_path = _path(input_config["historical_weights"])
    intervals_path = _path(input_config["industry_intervals"])
    sector_path = _path(input_config["sector_point_in_time_panel"])
    etf_path = _path(input_config["etf_total_return_daily"])
    constituent = pd.read_parquet(constituent_path)
    weights = pd.read_parquet(weights_path)
    intervals = pd.read_parquet(intervals_path)
    sector = pd.read_parquet(sector_path)
    etf = pd.read_parquet(etf_path)
    observation = determine_observation_state(
        config["prediction_date"],
        config["entry_date"],
        schedule,
        calendar,
        constituent,
        etf,
    )
    preview = build_forward_evaluation_report(
        config,
        rules,
        safety,
        calendar,
        observation,
        frozen_result,
        weights,
        intervals,
        constituent,
        sector,
        etf,
        runtime_sources={},
    )

    frozen_records = []
    for relative in sorted(FROZEN_FILES):
        path = _path(relative)
        if not path.is_file():
            raise ForwardEvaluationError(f"待冻结文件不存在：{relative}")
        frozen_records.append(
            {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size}
        )
    baseline_inputs = {
        "trading_calendars": [
            {
                "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                "sha256_at_freeze": file_sha256(path),
                "rows_at_freeze": int(len(frame)),
                "columns": list(frame.columns),
            }
            for path, frame in zip(calendar_paths, calendar_frames, strict=True)
        ],
        "constituent_total_return_daily": _frame_record(constituent_path, constituent),
        "historical_weights": _frame_record(weights_path, weights),
        "industry_intervals": _frame_record(intervals_path, intervals),
        "sector_point_in_time_panel": _frame_record(sector_path, sector),
        "etf_total_return_daily": _frame_record(etf_path, etf),
    }
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.date().isoformat() != config["prediction_date"]:
        raise ForwardEvaluationError("评价协议必须在首个前瞻观察日前冻结")
    manifest = {
        "manifest_version": "INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_MANIFEST",
        "status": "FROZEN_BEFORE_FIRST_FORWARD_OBSERVATION",
        "frozen_at": now.isoformat(timespec="seconds"),
        "prediction_date": config["prediction_date"],
        "entry_date": config["entry_date"],
        "original_prediction_manifest": {
            "path": frozen["manifest"],
            "sha256": file_sha256(_path(frozen["manifest"])),
        },
        "frozen_files": frozen_records,
        "mutable_outcome_input_baseline": baseline_inputs,
        "resolved_schedule_at_freeze": schedule,
        "preview": {
            "observation_date": preview["observation_date"],
            "primary_state": preview["primary_state"],
            "partial_return_output": False,
        },
        "prohibitions": {
            "original_prediction_rewrite": False,
            "partial_horizon_return_output": False,
            "missing_component_reweighting": False,
            "original_no_view_upgrade": False,
            "position_mapping": False,
            "order_generation": False,
            "broker_connection": False,
            "live_trading": False,
        },
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(_git_text("status", "--porcelain")),
        },
    }
    manifest_path = _path(config["outputs"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"前瞻评价协议已冻结：{manifest_path}")
    print(f"主要状态预览：{preview['primary_state']}")
    print(f"60日成熟日：{schedule[0]['maturity_date']}")
    print(f"120日日历状态：{schedule[1]['calendar_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
