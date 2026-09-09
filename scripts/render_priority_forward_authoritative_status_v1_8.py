"""生成与 V1.9 冻结运行时和直接采集回执一致的权威状态 V1.8。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from scripts.run_priority_forward_codex_automation_v1_9 import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/priority_forward_authoritative_status_v1_8.yaml"
EXPECTED_STREAM_IDS = [
    "A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1",
    "PRIMARY_MARKET_PCF_IOPV",
]
ALLOWED_COLLECTION_STATUSES = {
    "SUCCESS",
    "NO_COLLECTION",
    "PROGRAM_FAILED",
    "EXTERNAL_FREE_SOURCE_FAILED",
    "COMPLETE_QUALITY_DAY",
    "COLLECTED_INCOMPLETE_QUALITY_DAY",
    "SKIPPED_NON_TRADING_DAY",
    "FAILED_NO_CURRENT_TRADING_DAY_DATA",
    "PROGRAM_FAILED_INVALID_READINESS",
    "PROGRAM_FAILED_MISSING_TRADING_CALENDAR",
    "COMPLETED_OR_MARKET_CLOSED",
}


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"权威状态输入缺失：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _timestamp(payload: dict[str, Any]) -> str:
    return str(
        payload.get("ended_at")
        or payload.get("generated_at")
        or payload.get("started_at")
        or ""
    )


def select_latest_direct_task_receipt(
    receipt_globs: list[str],
) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    for pattern in receipt_globs:
        for path in sorted(ROOT.glob(pattern)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = _json(path)
            if payload.get("immutable_receipt") is not True:
                raise ValueError(f"PCF/IOPV 直接回执不是不可变证据：{path}")
            if payload.get("task_name") != "Codex-510300-Primary-Market-Collector":
                raise ValueError(f"PCF/IOPV 直接回执任务身份错配：{path}")
            declared = _path(str(payload.get("receipt_file") or ""))
            if declared.resolve() != resolved:
                raise ValueError(
                    "PCF/IOPV 直接回执声明路径错配："
                    f"actual={path}, declared={declared}"
                )
            candidates.append((path, payload))
    if not candidates:
        raise ValueError("没有 PCF/IOPV 不可变直接采集回执")
    return max(candidates, key=lambda item: (_timestamp(item[1]), item[0].as_posix()))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("权威状态 V1.8 配置顶层必须是对象")
    contract = config["contract"]
    streams = contract["active_research_streams"]
    if len(streams) != int(contract["maximum_active_research_streams"]):
        raise ValueError("活动研究流数量必须等于冻结上限")
    if [item["id"] for item in streams] != EXPECTED_STREAM_IDS:
        raise ValueError("活动研究流与最终决议不一致")
    if sum(int(item["resource_percent"]) for item in streams) != 100:
        raise ValueError("活动研究流资源比例必须合计 100%")
    if contract["only_new_frozen_test"] != EXPECTED_STREAM_IDS[0]:
        raise ValueError("唯一新冻结测试必须是沪深300官方调入候选")
    if contract.get("original_90_day_decision") != "RUN_ONE_NEW_FROZEN_TEST":
        raise ValueError("原始90日决策必须原样保留")
    if contract.get("decision") != "STOP_NO_HISTORICAL_RETURN_EVALUATION":
        raise ValueError("数据门失败后必须停止历史收益评价")
    if int(contract.get("historical_return_attempts_allowed", -1)) != 0:
        raise ValueError("数据门失败后历史收益评价许可必须为0")
    authorization = config["execution_authorization"]
    if not authorization["research_only"] or any(
        authorization[key]
        for key in (
            "shadow_enabled",
            "position_mapping_enabled",
            "order_generation_enabled",
            "broker_connection_enabled",
            "live_trading_enabled",
        )
    ):
        raise ValueError("V1.8 必须保持研究专用且全部执行权限关闭")
    globs = config["inputs"].get("primary_market_task_receipt_globs")
    if not isinstance(globs, list) or not globs:
        raise ValueError("V1.8 缺少 PCF/IOPV 不可变回执范围")
    return config


def _validate_csi_terminal(
    config: dict[str, Any],
) -> dict[str, Any]:
    contract = config["contract"]
    inputs = config["inputs"]
    paths = {
        "source_admission": _path(inputs["csi300_source_admission"]),
        "data_admission": _path(inputs["csi300_data_admission"]),
        "pre_return_freeze": _path(inputs["csi300_pre_return_freeze"]),
    }
    payloads = {name: _json(path) for name, path in paths.items()}
    strategy_id = str(contract["only_new_frozen_test"])
    for name, payload in payloads.items():
        if payload.get("strategy_id") != strategy_id:
            raise ValueError(f"CSI300 {name} 策略身份错配")
    data_admission = payloads["data_admission"]
    freeze = payloads["pre_return_freeze"]
    if freeze.get("status") != "NO_VIEW_DATA_CONTRACT_FAILED":
        raise ValueError("CSI300 当前终局必须是数据合同失败 NO_VIEW")
    if data_admission.get("status") != freeze.get("status"):
        raise ValueError("CSI300 D4/D5 准入与最终冻结状态不一致")
    if freeze.get("return_evaluation") != "NOT_ALLOWED":
        raise ValueError("CSI300 数据门失败后不得允许收益评价")
    if data_admission.get("return_evaluation") != "NOT_ALLOWED":
        raise ValueError("CSI300 D4/D5 准入没有关闭收益评价")
    if data_admission.get("source_rescue_allowed") is not False:
        raise ValueError("CSI300 数据门失败后不得开放来源救援")
    failed_gates = list(freeze.get("failed_data_gates") or [])
    if failed_gates != ["D4", "D5"]:
        raise ValueError(f"CSI300 最终失败门不一致：{failed_gates}")
    attempts_allowed = int(freeze.get("historical_return_attempts_allowed", -1))
    attempts_used = int(freeze.get("historical_return_attempts_used", -1))
    if attempts_allowed != 0 or attempts_used != 0:
        raise ValueError("CSI300 数据门失败后历史收益检验次数必须为0")
    return {
        "id": strategy_id,
        "priority": 1,
        "resource_percent": 70,
        "status": str(freeze["status"]),
        "return_evaluation": str(freeze["return_evaluation"]),
        "terminal": True,
        "additional_historical_work_allowed": False,
        "source_rescue_allowed": False,
        "historical_return_attempts_allowed": attempts_allowed,
        "historical_return_attempts_used": attempts_used,
        "failed_data_gates": failed_gates,
        "evidence": {name: _evidence(path) for name, path in paths.items()},
    }


def build_report(config: dict[str, Any], generated_at: datetime) -> dict[str, Any]:
    contract = config["contract"]
    inputs = config["inputs"]
    readiness_path = _path(inputs["primary_market_readiness"])
    readiness = _json(readiness_path)
    latest_path, latest = select_latest_direct_task_receipt(
        [str(item) for item in inputs["primary_market_task_receipt_globs"]]
    )
    if int(latest.get("full_coverage_days", -1)) != int(
        readiness["full_coverage_days"]
    ):
        raise ValueError("最新直接回执与 readiness 的完整质量日数量不一致")
    if str(latest.get("latest_observed_trade_date")) != str(
        readiness["last_observed_trade_date"]
    ):
        raise ValueError("最新直接回执与 readiness 的最后观测交易日不一致")
    collection_status = str(latest.get("collection_status") or "")
    if collection_status not in ALLOWED_COLLECTION_STATUSES:
        raise ValueError(f"未知 PCF/IOPV collection_status：{collection_status}")
    task_exit_code = int(latest.get("task_exit_code", -1))
    quality_day_counted = (
        collection_status == "COMPLETE_QUALITY_DAY" and task_exit_code == 0
    )

    manifest_path = _path(inputs["priority_forward_manifest"])
    manifest_hash = _sha256(manifest_path)
    manifest_verification = verify_manifest(manifest_path, manifest_hash)
    manifest = _json(manifest_path)
    expected_patch = str(inputs["expected_runtime_patch_id"])
    if manifest.get("runtime_patch_id") != expected_patch:
        raise ValueError("权威状态与 V1.9 运行补丁标识不一致")

    log_value = str(latest.get("log_file") or "")
    log_evidence = _evidence(_path(log_value)) if log_value else None
    runtime_patch_id = str(latest.get("runtime_patch_id") or "")
    deployment_status = (
        "ACTIVE_FIRST_ELIGIBLE_RUN_OBSERVED"
        if runtime_patch_id == expected_patch
        else str(manifest["deployment_status_until_first_run"])
    )
    csi_stream = _validate_csi_terminal(config)
    pcf_stream = {
        "id": "PRIMARY_MARKET_PCF_IOPV",
        "priority": 2,
        "resource_percent": 30,
        "status": str(readiness["status"]),
        "stage": str(readiness["stage"]),
        "deployment_status": deployment_status,
        "latest_observed_trade_date": str(readiness["last_observed_trade_date"]),
        "observed_trading_days": int(readiness["observed_trading_days"]),
        "full_quality_days": int(readiness["full_coverage_days"]),
        "quality_day_gate": int(readiness["minimum_full_coverage_days"]),
        "feature_freeze_gate": int(readiness["recommended_full_coverage_days"]),
        "first_unseen_gate": int(
            readiness["first_unseen_evaluation_full_coverage_days"]
        ),
        "replication_gate": int(readiness["replication_full_coverage_days"]),
        "latest_task_authority": {
            "authority_kind": "DIRECT_COLLECTOR_IMMUTABLE_RECEIPT",
            "path": latest_path.relative_to(ROOT).as_posix(),
            "receipt_id": latest["receipt_id"],
            "ended_at": latest.get("ended_at"),
            "runtime_patch_id": runtime_patch_id,
            "run_status": latest.get("run_status"),
            "collection_status": collection_status,
            "task_exit_code": task_exit_code,
            "error": latest.get("error"),
            "quality_day_counted": quality_day_counted,
            "receipt_evidence": _evidence(latest_path),
            "log_evidence": log_evidence,
        },
        "readiness_evidence": _evidence(readiness_path),
    }
    execution = config["execution_authorization"]
    return {
        "schema_version": "1.8.0",
        "report_version": config["version"],
        "generated_at": generated_at.isoformat(),
        "overall_research_status": contract["overall_research_status"],
        "original_90_day_decision": contract["original_90_day_decision"],
        "decision": contract["decision"],
        "only_new_frozen_test": contract["only_new_frozen_test"],
        "active_research_streams": [csi_stream, pcf_stream],
        "frozen_priority_forward_manifest": {
            **_evidence(manifest_path),
            "manifest_id": manifest["manifest_id"],
            "runtime_patch_id": manifest["runtime_patch_id"],
            "verification_status": manifest_verification["status"],
            "tracked_file_count": manifest_verification["tracked_file_count"],
        },
        "status_semantics": {
            "task_exit_code_0": "DIRECT_RECEIPT_DECIDES_QUALITY_STATUS",
            "task_exit_code_1": "PROGRAM_FAILED",
            "task_exit_code_2": "NO_COLLECTION_OR_MISSED_WINDOW",
            "task_exit_code_3": "EXTERNAL_FREE_SOURCE_FAILED",
            "already_attempted": "ORCHESTRATION_DEDUPLICATION_NOT_RESEARCH_SUCCESS",
            "missing_latest_status": "NO_SIGNAL_NO_REPORT_NO_ORDER",
            "read_only_probe": "NOT_A_QUALITY_DAY",
        },
        "execution_authorization": {
            "research_only": bool(execution["research_only"]),
            "shadow_enabled": bool(execution["shadow_enabled"]),
            "position_mapping_enabled": bool(execution["position_mapping_enabled"]),
            "order_generation_enabled": bool(execution["order_generation_enabled"]),
            "broker_connection_enabled": bool(execution["broker_connection_enabled"]),
            "live_trading_enabled": bool(execution["live_trading_enabled"]),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    csi = report["active_research_streams"][0]
    pcf = report["active_research_streams"][1]
    latest = pcf["latest_task_authority"]
    manifest = report["frozen_priority_forward_manifest"]
    return "\n".join(
        [
            "# 优先前瞻研究权威状态 V1.8",
            "",
            f"- 研究结论：`{report['overall_research_status']}`",
            f"- 当前决策：`{report['decision']}`",
            f"- 冻结运行清单：`{manifest['manifest_id']}`（哈希已验证）。",
            "",
            "## 研究流",
            "",
            (
                "1. 沪深300官方调入候选："
                f"`{csi['status']}`；失败门 `{','.join(csi['failed_data_gates'])}`；"
                "收益评价 `NOT_ALLOWED`。"
            ),
            (
                "2. PCF/IOPV："
                f"{pcf['full_quality_days']}/{pcf['quality_day_gate']} 个完整质量日；"
                f"部署状态 `{pcf['deployment_status']}`。"
            ),
            "",
            "## PCF/IOPV 最新直接回执",
            "",
            f"- 回执：`{latest['path']}`",
            f"- 状态：`{latest['collection_status']}`；退出码 `{latest['task_exit_code']}`。",
            f"- 本次是否计质量日：`{str(latest['quality_day_counted']).lower()}`。",
            "",
            "研究、Shadow、仓位映射、订单、券商连接和实盘继续分离；当前只有研究权限。",
            "",
        ]
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成优先前瞻研究权威状态 V1.8")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config_path = arguments.config if arguments.config.is_absolute() else ROOT / arguments.config
    config = load_config(config_path)
    generated_at = datetime.now(ZoneInfo(str(config["timezone"])))
    report = build_report(config, generated_at)
    _atomic_text(
        _path(config["outputs"]["json"]),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(_path(config["outputs"]["markdown"]), render_markdown(report))
    pcf = report["active_research_streams"][1]
    latest = pcf["latest_task_authority"]
    print(f"权威状态：{report['overall_research_status']}")
    print(f"PCF/IOPV 部署状态：{pcf['deployment_status']}")
    print(
        "PCF/IOPV 最新直接状态："
        f"{latest['collection_status']}，退出码 {latest['task_exit_code']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
