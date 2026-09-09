"""V1_0_1数据预热修正版；模型参数与V1逐项相同。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

import macd_breadth_downside_preflight_v1 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macd_breadth_downside_preflight_v1_0_1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_macd_breadth_downside_preflight_v1_0_1_manifest.json"

ContractError = base.ContractError
sha256_file = base.sha256_file


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _load_overlay() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise ContractError(f"缺少修正协议：{CONFIG_PATH}")
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("修正协议必须是YAML对象")
    return payload


def load_config() -> dict[str, Any]:
    overlay = _load_overlay()
    base_spec = overlay["base"]
    base_config_path = _project_path(base_spec["config_path"])
    base_manifest_path = _project_path(base_spec["manifest_path"])
    if sha256_file(base_config_path) != base_spec["config_sha256"]:
        raise ContractError("V1基础协议哈希漂移")
    if sha256_file(base_manifest_path) != base_spec["manifest_sha256"]:
        raise ContractError("V1冻结清单哈希漂移")
    config = deepcopy(base.load_config(base_config_path))
    correction = overlay["correction"]
    replacement = overlay["input_replacement"]
    overrides = overlay["status_overrides"]
    if correction["prior_candidate_outcomes_read"] is not False:
        raise ContractError("只有未读取任何候选结果的数据失败才允许本修正")
    immutable_flags = [
        "evaluation_window_changed",
        "feature_definition_changed",
        "event_definition_changed",
        "direction_changed",
        "gate_changed",
        "bootstrap_changed",
    ]
    if any(correction[key] is not False for key in immutable_flags):
        raise ContractError("修正协议声明改变了冻结研究参数")
    replacement_path = _project_path(replacement["replacement_path"])
    if sha256_file(replacement_path) != replacement["replacement_sha256"]:
        raise ContractError("广度预热修复文件哈希不一致")
    config["protocol"]["project_id"] = correction["project_id"]
    config["protocol"]["state"] = overrides["protocol_state"]
    config["protocol"]["evidence_class"] = overrides["evidence_class"]
    key = replacement["key"]
    if config["inputs"][key]["path"] != replacement["prior_path"]:
        raise ContractError("V1基础输入路径与修正声明不一致")
    config["inputs"][key]["path"] = replacement["replacement_path"]
    config["adjudication"]["pass_status"] = overrides["pass_status"]
    config["adjudication"]["fail_status"] = overrides["fail_status"]
    config["paths"] = deepcopy(overlay["paths"])
    config["correction"] = deepcopy(overlay)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    overlay = config["correction"]
    correction = overlay["correction"]
    if config["protocol"]["project_id"] != correction["project_id"]:
        raise ContractError("V1_0_1 project_id不一致")
    if config["dates"]["evaluation_start"] != "2021-08-12":
        raise ContractError("V1_0_1评价起点发生变化")
    if config["dates"]["evaluation_end"] != "2026-08-12":
        raise ContractError("V1_0_1评价终点发生变化")
    if config["features"] != base.load_config()["features"]:
        raise ContractError("V1_0_1特征参数与V1不一致")
    if config["outcomes"] != base.load_config()["outcomes"]:
        raise ContractError("V1_0_1结果定义与V1不一致")
    if config["event_sampling"] != base.load_config()["event_sampling"]:
        raise ContractError("V1_0_1事件抽样与V1不一致")
    if config["bootstrap"] != base.load_config()["bootstrap"]:
        raise ContractError("V1_0_1 Bootstrap与V1不一致")
    if config["gates"] != base.load_config()["gates"]:
        raise ContractError("V1_0_1验收门槛与V1不一致")
    if config["scope"] != base.load_config()["scope"]:
        raise ContractError("V1_0_1资产与执行边界发生变化")
    if config["governance"] != base.load_config()["governance"]:
        raise ContractError("V1_0_1治理边界发生变化")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("V1_0_1尚未冻结：缺少manifest")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("V1_0_1 manifest project_id不一致")
    if manifest.get("state") != "FROZEN_BEFORE_FIRST_RESULT":
        raise ContractError("V1_0_1 manifest未处于冻结状态")
    mismatches: dict[str, dict[str, str | None]] = {}
    for section in ("tracked_files", "input_files"):
        records = manifest.get(section)
        if not isinstance(records, dict) or not records:
            raise ContractError(f"V1_0_1 manifest缺少{section}")
        for relative, expected in records.items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else None
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"V1_0_1冻结文件哈希漂移：{mismatches}")
    return {
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "tracked_file_count": len(manifest["tracked_files"]),
        "input_file_count": len(manifest["input_files"]),
        "hash_mismatches": {},
    }


def run_preflight(*, write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    market, dividends, breadth = base.load_inputs(config)
    features, distributions = base.build_daily_features(
        market, dividends, breadth, config
    )
    events = base.select_events(features, config)
    evaluation = base.evaluate_preflight(features, events, config)
    report = base.build_report(
        config, manifest, features, events, distributions, evaluation
    )
    correction = config["correction"]
    report["correction"] = {
        "type": correction["correction"]["type"],
        "reason": correction["correction"]["reason"],
        "prior_run_status": correction["correction"]["prior_run_status"],
        "prior_candidate_outcomes_read": False,
        "only_change": correction["correction"]["only_change"],
        "remediation_report": correction["input_replacement"][
            "remediation_report"
        ],
    }
    if write:
        paths = config["paths"]
        base._atomic_parquet(_project_path(paths["feature_table"]), features)
        base._atomic_parquet(_project_path(paths["event_table"]), events)
        base._atomic_json(_project_path(paths["result_json"]), report)
        base._atomic_text(
            _project_path(paths["result_markdown"]), base.render_markdown(report)
        )
    return report


__all__ = [
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "ContractError",
    "load_config",
    "run_preflight",
    "sha256_file",
    "validate_config",
    "validate_manifest",
]
