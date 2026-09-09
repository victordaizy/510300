"""以精确恢复覆盖层运行行业预期差 V1 追加式前瞻评价。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    render_forward_evaluation_markdown,
    resolve_maturity_schedule,
    validate_evaluation_config,
    verify_hash_manifest,
)
from research.industry_expectation_gap_frozen_input_recovery import (  # noqa: E402
    FrozenInputRecoveryError,
    validate_recovery_config,
    verify_manifest_with_declared_recoveries,
)


RECOVERY_CONFIG_PATH = (
    WORKSPACE_ROOT
    / "config"
    / "industry_expectation_gap_forward_evaluation_v1_1_recovery.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ForwardEvaluationError(f"无法读取 YAML：{path}") from exc
    if not isinstance(value, dict):
        raise ForwardEvaluationError(f"YAML 顶层必须是对象：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ForwardEvaluationError(f"无法读取 JSON：{path}") from exc
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


def _relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def _read_calendars(config: dict[str, Any]) -> tuple[list[Path], pd.DatetimeIndex]:
    pattern = config["mutable_outcome_inputs"]["trading_calendar_glob"]
    paths = sorted(WORKSPACE_ROOT.glob(pattern))
    if not paths:
        raise ForwardEvaluationError(f"没有匹配交易日历：{pattern}")
    return paths, load_trading_calendar([pd.read_csv(path) for path in paths])


def _verify_calendar_metadata(config: dict[str, Any], calendar_paths: list[Path]) -> list[Path]:
    pattern = config["mutable_outcome_inputs"]["trading_calendar_metadata_glob"]
    metadata_paths = sorted(WORKSPACE_ROOT.glob(pattern))
    by_stem = {
        path.name.replace(".metadata.json", ""): path for path in metadata_paths
    }
    for calendar_path in calendar_paths:
        metadata_path = by_stem.get(calendar_path.stem)
        if metadata_path is None:
            raise ForwardEvaluationError(f"交易日历缺少元数据：{calendar_path}")
        metadata = _load_json(metadata_path)
        if metadata.get("status") != "PASS":
            raise ForwardEvaluationError(f"交易日历元数据未通过：{metadata_path}")
        cross_check = metadata.get("secondary_cross_check", {})
        if cross_check.get("date_set_matches") is not True:
            raise ForwardEvaluationError(f"交易日历独立交叉核验未通过：{metadata_path}")
        if metadata.get("sha256") != file_sha256(calendar_path):
            raise ForwardEvaluationError(f"交易日历与元数据哈希不一致：{calendar_path}")
    return metadata_paths


def _source_record(path: Path, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if frame is not None:
        record["rows"] = int(len(frame))
        record["columns"] = list(frame.columns)
        if "date" in frame.columns and len(frame):
            dates = pd.to_datetime(frame["date"], errors="coerce")
            record["date_min"] = dates.min().date().isoformat()
            record["date_max"] = dates.max().date().isoformat()
    return record


def _write_append_only(path: Path, content: str) -> str:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise ForwardEvaluationError(f"追加式输出已存在且内容不同，禁止覆盖：{path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "CREATED"


def _verify_named_file(relative: str, expected_hash: str, label: str) -> dict[str, Any]:
    path = _path(relative)
    if not path.is_file():
        raise FrozenInputRecoveryError(f"{label}不存在：{relative}")
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash.lower():
        raise FrozenInputRecoveryError(f"{label}哈希不一致：{relative}")
    return {"path": relative, "sha256": actual_hash, "bytes": path.stat().st_size}


def main() -> int:
    recovery_config = _load_yaml(RECOVERY_CONFIG_PATH)
    validate_recovery_config(recovery_config)

    outputs = recovery_config["outputs"]
    verify_hash_manifest(WORKSPACE_ROOT, outputs["manifest"])
    parent = recovery_config["parent_evaluation"]
    _verify_named_file(parent["config"], parent["config_sha256"], "原评价配置")
    verify_hash_manifest(
        WORKSPACE_ROOT,
        parent["manifest"],
        parent["manifest_sha256"],
    )
    parent_config = _load_yaml(_path(parent["config"]))
    rules, safety = validate_evaluation_config(parent_config)

    frozen = recovery_config["frozen_prediction"]
    prediction_verification = verify_manifest_with_declared_recoveries(
        WORKSPACE_ROOT,
        manifest_relative_path=frozen["manifest"],
        expected_manifest_sha256=frozen["manifest_sha256"],
        recoveries=recovery_config["recoveries"],
    )
    result_record = _verify_named_file(
        frozen["result"], frozen["result_sha256"], "原预测结果"
    )
    ledger_record = _verify_named_file(
        frozen["ledger"], frozen["ledger_sha256"], "原行业预测账本"
    )
    frozen_result = _load_json(_path(frozen["result"]))

    calendar_paths, calendar = _read_calendars(parent_config)
    metadata_paths = _verify_calendar_metadata(parent_config, calendar_paths)
    schedule = resolve_maturity_schedule(
        calendar,
        parent_config["entry_date"],
        rules.horizons_trading_days,
        parent_config.get("pre_resolved_maturity_dates"),
    )

    inputs = parent_config["mutable_outcome_inputs"]
    constituent_path = _path(inputs["constituent_total_return_daily"])
    weights_path = _path(inputs["historical_weights"])
    intervals_path = _path(inputs["industry_intervals"])
    sector_path = _path(inputs["sector_point_in_time_panel"])
    etf_path = _path(inputs["etf_total_return_daily"])
    constituent = pd.read_parquet(constituent_path)
    weights = pd.read_parquet(weights_path)
    intervals = pd.read_parquet(intervals_path)
    sector = pd.read_parquet(sector_path)
    etf = pd.read_parquet(etf_path)
    observation = determine_observation_state(
        parent_config["prediction_date"],
        parent_config["entry_date"],
        schedule,
        calendar,
        constituent,
        etf,
    )
    runtime_sources = {
        "trading_calendars": [_source_record(path) for path in calendar_paths],
        "trading_calendar_metadata": [_source_record(path) for path in metadata_paths],
        "constituent_total_return_daily": _source_record(constituent_path, constituent),
        "historical_weights": _source_record(weights_path, weights),
        "industry_intervals": _source_record(intervals_path, intervals),
        "sector_point_in_time_panel": _source_record(sector_path, sector),
        "etf_total_return_daily": _source_record(etf_path, etf),
        "frozen_prediction_manifest": {
            "path": frozen["manifest"],
            "sha256": file_sha256(_path(frozen["manifest"])),
        },
    }
    report = build_forward_evaluation_report(
        parent_config,
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
        runtime_sources,
    )
    report["evaluation_execution_version"] = recovery_config["version"]
    report["recovery_addendum"] = {
        "status": recovery_config["status"],
        "purpose": recovery_config["purpose"],
        "manifest": {
            "path": outputs["manifest"],
            "sha256": file_sha256(_path(outputs["manifest"])),
        },
        "parent_evaluation_manifest": {
            "path": parent["manifest"],
            "sha256": parent["manifest_sha256"],
        },
        "prediction_verification": prediction_verification,
        "frozen_result": result_record,
        "frozen_ledger": ledger_record,
        "prediction_change": False,
        "rule_change": False,
        "threshold_change": False,
        "current_operational_readiness_replaced": False,
    }

    output_directory = _path(outputs["directory"])
    stem = f"{outputs['stem_prefix']}_{report['observation_date']}"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_forward_evaluation_markdown(report)
    markdown_text += "\n## 冻结输入恢复\n\n"
    markdown_text += "- 执行层：`INDUSTRY_EXPECTATION_GAP_FORWARD_EVALUATION_V1_1`\n"
    markdown_text += "- 仅恢复原冻结输入身份；预测、规则、阈值均未变化。\n"
    markdown_text += "- 当前 PCF/IOPV 就绪报告未被替换。\n"
    json_state = _write_append_only(json_path, json_text)
    markdown_state = _write_append_only(markdown_path, markdown_text)

    print(f"前瞻评价状态：{report['primary_state']}")
    print(f"观察日：{report['observation_date']}")
    print(f"恢复冻结对象：{prediction_verification['recovered_target_count']}")
    print(f"JSON：{json_path}（{json_state}）")
    print(f"Markdown：{markdown_path}（{markdown_state}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
