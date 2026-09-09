"""生成与当前唯一新测试及 PCF/IOPV 直接回执一致的权威状态 V1.7。"""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "priority_forward_authoritative_status_v1_7.yaml"


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


def _json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"权威状态输入缺失：{path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return payload


def _timestamp(payload: dict[str, Any]) -> str:
    return str(payload.get("ended_at") or payload.get("generated_at") or payload.get("started_at") or "")


def select_latest_direct_task_status(
    candidates: list[tuple[Path, dict[str, Any]]],
) -> tuple[Path, dict[str, Any]]:
    """按直接采集结束时间选最新回执，同一 receipt_id 优先不可变回执目录。"""

    if not candidates:
        raise ValueError("没有 PCF/IOPV 直接采集状态候选")
    by_receipt_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, payload in candidates:
        receipt_id = str(payload.get("receipt_id") or "")
        if not receipt_id:
            raise ValueError(f"PCF/IOPV 状态缺少 receipt_id：{path}")
        if payload.get("immutable_receipt") is not True:
            raise ValueError(f"PCF/IOPV 状态不是不可变回执：{path}")
        if payload.get("task_name") != "Codex-510300-Primary-Market-Collector":
            raise ValueError(f"PCF/IOPV 状态任务身份错配：{path}")
        existing = by_receipt_id.get(receipt_id)
        is_receipt_path = "primary_market_task_runs_v1_2_1" in path.as_posix()
        existing_is_receipt_path = bool(
            existing and "primary_market_task_runs_v1_2_1" in existing[0].as_posix()
        )
        if existing is None or (is_receipt_path and not existing_is_receipt_path):
            by_receipt_id[receipt_id] = (path, payload)
    return max(
        by_receipt_id.values(),
        key=lambda item: (_timestamp(item[1]), item[0].as_posix()),
    )


def _task_candidates(current: Path, receipt_glob: str) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if current.is_file():
        current_payload = _json(current, required=True)
        assert current_payload is not None
        candidates.append((current, current_payload))
    for path in sorted(ROOT.glob(receipt_glob)):
        payload = _json(path, required=True)
        assert payload is not None
        candidates.append((path, payload))
    return candidates


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("权威状态 V1.7 配置顶层必须是对象")
    contract = config["contract"]
    streams = contract["active_research_streams"]
    if len(streams) != int(contract["maximum_active_research_streams"]):
        raise ValueError("活动研究流数量必须等于冻结上限")
    if sum(int(item["resource_percent"]) for item in streams) != 100:
        raise ValueError("活动研究流资源比例必须合计 100%")
    if [item["id"] for item in streams] != [
        "A_SHARE_HS_CSI300_OFFICIAL_ADDITION_FORCED_DEMAND_V1",
        "PRIMARY_MARKET_PCF_IOPV",
    ]:
        raise ValueError("活动研究流与最终决议不一致")
    if contract["only_new_frozen_test"] != streams[0]["id"]:
        raise ValueError("唯一新冻结测试必须是沪深300官方调入候选")
    if contract.get("original_90_day_decision") != "RUN_ONE_NEW_FROZEN_TEST":
        raise ValueError("原始90日决策必须原样保留")
    if contract.get("decision") != "STOP_NO_HISTORICAL_RETURN_EVALUATION":
        raise ValueError("数据门失败后的当前决策必须停止历史收益评价")
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
        raise ValueError("V1.7 必须保持研究专用且全部执行权限关闭")
    return config


def build_report(config: dict[str, Any], generated_at: datetime) -> dict[str, Any]:
    contract = config["contract"]
    inputs = config["inputs"]
    readiness_path = _path(inputs["primary_market_readiness"])
    readiness = _json(readiness_path, required=True)
    assert readiness is not None

    current_path = _path(inputs["primary_market_current_task_status"])
    latest_task_path, latest_task = select_latest_direct_task_status(
        _task_candidates(current_path, inputs["primary_market_task_receipt_glob"])
    )
    receipt_file = _path(str(latest_task["receipt_file"]))
    if receipt_file.resolve() != latest_task_path.resolve():
        raise ValueError(
            "最新状态没有落到声明的不可变直接回执："
            f"selected={latest_task_path}, declared={receipt_file}"
        )
    if int(latest_task.get("full_coverage_days", -1)) != int(readiness["full_coverage_days"]):
        raise ValueError("最新直接回执与 readiness 的完整质量日数量不一致")
    if str(latest_task.get("latest_observed_trade_date")) != str(readiness["last_observed_trade_date"]):
        raise ValueError("最新直接回执与 readiness 的最后观测交易日不一致")

    manifest_path = _path(inputs["priority_forward_manifest"])
    manifest_evidence = _evidence(manifest_path)
    expected_manifest_hash = str(inputs["priority_forward_manifest_expected_sha256"])
    if manifest_evidence["sha256"] != expected_manifest_hash:
        raise ValueError(
            "V1.8 冻结清单 SHA-256 不一致："
            f"expected={expected_manifest_hash}, actual={manifest_evidence['sha256']}"
        )

    log_path = _path(str(latest_task["log_file"]))
    receipt_evidence = _evidence(latest_task_path)
    log_evidence = _evidence(log_path)
    raw_response_path = latest_task.get("raw_response_path") or latest_task.get("response_path")
    raw_response_sha256 = latest_task.get("raw_response_sha256") or latest_task.get("response_sha256")
    raw_response_present = bool(raw_response_path and raw_response_sha256)

    collection_status = str(latest_task.get("collection_status") or "")
    if collection_status not in {
        "SUCCESS",
        "NO_COLLECTION",
        "PROGRAM_FAILED",
        "EXTERNAL_FREE_SOURCE_FAILED",
    }:
        raise ValueError(f"未知 PCF/IOPV collection_status：{collection_status}")
    quality_day_counted = collection_status == "SUCCESS" and int(latest_task.get("task_exit_code", -1)) == 0

    csi_admission_path = _path(inputs["csi300_source_admission"])
    csi_data_admission_path = _path(inputs["csi300_data_admission"])
    csi_freeze_path = _path(inputs["csi300_pre_return_freeze"])
    csi_admission = _json(csi_admission_path, required=True)
    csi_data_admission = _json(csi_data_admission_path, required=True)
    csi_freeze = _json(csi_freeze_path, required=True)
    assert csi_admission is not None
    assert csi_data_admission is not None
    assert csi_freeze is not None
    strategy_id = str(contract["only_new_frozen_test"])
    for label, payload in (
        ("来源准入", csi_admission),
        ("D4/D5准入", csi_data_admission),
        ("最终冻结", csi_freeze),
    ):
        if payload.get("strategy_id") != strategy_id:
            raise ValueError(f"{label}策略标识与唯一新冻结测试不一致")

    csi_status = str(csi_freeze.get("status") or "")
    csi_return_evaluation = str(csi_freeze.get("return_evaluation") or "")
    if csi_status != "NO_VIEW_DATA_CONTRACT_FAILED":
        raise ValueError("V1.7 当前权威状态必须吸收数据门失败终局")
    if csi_data_admission.get("status") != csi_status:
        raise ValueError("D4/D5准入报告与最终冻结状态不一致")
    if csi_return_evaluation != "NOT_ALLOWED":
        raise ValueError("数据门失败后不得允许收益评估")
    if csi_data_admission.get("return_evaluation") != csi_return_evaluation:
        raise ValueError("D4/D5准入报告与最终冻结收益权限不一致")
    if csi_data_admission.get("decision") != "STOP_NO_HISTORICAL_RETURN_EVALUATION":
        raise ValueError("D4/D5准入报告没有执行停止决策")
    if csi_data_admission.get("source_rescue_allowed") is not False:
        raise ValueError("数据门失败后不得开放来源救援")

    data_gate_summary = dict(csi_freeze.get("data_gate_summary") or {})
    failed_data_gates = list(csi_freeze.get("failed_data_gates") or [])
    if failed_data_gates != ["D4", "D5"]:
        raise ValueError(f"最终冻结失败门与当前终局不一致：{failed_data_gates}")
    if any(data_gate_summary.get(gate) != "PASS" for gate in ("D1", "D2", "D3", "D6", "D7", "D8")):
        raise ValueError("最终冻结的通过门状态不完整")
    if any(data_gate_summary.get(gate) != "FAIL" for gate in failed_data_gates):
        raise ValueError("最终冻结的失败门状态不完整")
    historical_attempts_allowed = int(csi_freeze.get("historical_return_attempts_allowed", -1))
    historical_attempts_used = int(csi_freeze.get("historical_return_attempts_used", -1))
    if historical_attempts_allowed != int(contract["historical_return_attempts_allowed"]):
        raise ValueError("权威状态配置与最终冻结的历史许可不一致")
    if historical_attempts_used != 0:
        raise ValueError("数据门失败前后均不得消耗历史收益检验次数")

    csi_protocol_manifest_path = _path(str(csi_freeze["protocol_manifest_path"]))
    csi_protocol_manifest_evidence = _evidence(csi_protocol_manifest_path)
    if csi_protocol_manifest_evidence["sha256"] != csi_freeze.get("protocol_manifest_sha256"):
        raise ValueError("最终冻结引用的协议清单哈希不一致")

    execution = config["execution_authorization"]
    return {
        "schema_version": "1.7.1",
        "report_version": config["version"],
        "generated_at": generated_at.isoformat(),
        "overall_research_status": contract["overall_research_status"],
        "original_90_day_decision": contract["original_90_day_decision"],
        "decision": contract["decision"],
        "only_new_frozen_test": contract["only_new_frozen_test"],
        "active_research_streams": [
            {
                "id": contract["active_research_streams"][0]["id"],
                "priority": 1,
                "resource_percent": 70,
                "status": csi_status,
                "return_evaluation": csi_return_evaluation,
                "terminal": True,
                "additional_historical_work_allowed": False,
                "historical_return_attempts_allowed": historical_attempts_allowed,
                "historical_return_attempts_used": historical_attempts_used,
                "failed_data_gates": failed_data_gates,
                "data_gate_summary": data_gate_summary,
                "source_rescue_allowed": False,
                "source_admission": _evidence(csi_admission_path),
                "data_admission": _evidence(csi_data_admission_path),
                "pre_return_freeze": _evidence(csi_freeze_path),
                "protocol_manifest": {
                    **csi_protocol_manifest_evidence,
                    "expected_sha256": csi_freeze["protocol_manifest_sha256"],
                    "hash_verified": True,
                },
            },
            {
                "id": "PRIMARY_MARKET_PCF_IOPV",
                "priority": 2,
                "resource_percent": 30,
                "status": str(readiness["status"]),
                "stage": str(readiness["stage"]),
                "latest_observed_trade_date": str(readiness["last_observed_trade_date"]),
                "observed_trading_days": int(readiness["observed_trading_days"]),
                "full_quality_days": int(readiness["full_coverage_days"]),
                "quality_day_gate": int(readiness["minimum_full_coverage_days"]),
                "feature_freeze_gate": int(readiness["recommended_full_coverage_days"]),
                "first_unseen_gate": int(readiness["first_unseen_evaluation_full_coverage_days"]),
                "replication_gate": int(readiness["replication_full_coverage_days"]),
                "latest_task_authority": {
                    "authority_kind": "DIRECT_COLLECTOR_IMMUTABLE_RECEIPT",
                    "path": latest_task_path.relative_to(ROOT).as_posix(),
                    "receipt_id": latest_task["receipt_id"],
                    "ended_at": latest_task.get("ended_at"),
                    "run_status": latest_task.get("run_status"),
                    "collection_status": collection_status,
                    "task_exit_code": int(latest_task.get("task_exit_code")),
                    "error": latest_task.get("error"),
                    "quality_day_counted": quality_day_counted,
                    "receipt_evidence": receipt_evidence,
                    "log_evidence": log_evidence,
                    "raw_response_present": raw_response_present,
                    "raw_response_path": raw_response_path,
                    "raw_response_sha256": raw_response_sha256,
                },
                "readiness_evidence": _evidence(readiness_path),
            },
        ],
        "frozen_v1_8_manifest": {
            **manifest_evidence,
            "expected_sha256": expected_manifest_hash,
            "hash_verified": True,
        },
        "status_semantics": {
            "task_exit_code_0": "COLLECTION_SUCCESS_ONLY_IF_DIRECT_RECEIPT_SAYS_SUCCESS",
            "task_exit_code_1": "PROGRAM_FAILED",
            "task_exit_code_2": "NO_COLLECTION",
            "task_exit_code_3": "EXTERNAL_FREE_SOURCE_FAILED",
            "already_attempted": "ORCHESTRATION_DEDUPLICATION_NOT_RESEARCH_SUCCESS",
            "missing_latest_status": "NO_SIGNAL_NO_REPORT_NO_ORDER",
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
    return "\n".join(
        [
            "# 优先前瞻研究权威状态 V1.7",
            "",
            f"- 研究结论：`{report['overall_research_status']}`",
            f"- 原始90日决策：`{report['original_90_day_decision']}`",
            f"- 当前终局：`{report['decision']}`",
            f"- 唯一新冻结测试：`{report['only_new_frozen_test']}`",
            "",
            "## 两条原计划研究流",
            "",
            (
                "1. 沪深300官方调入候选（原计划70%）："
                f"`{csi['status']}`；失败门 `{','.join(csi['failed_data_gates'])}`；"
                f"收益评估 `{csi['return_evaluation']}`；历史许可 "
                f"{csi['historical_return_attempts_allowed']} 次、已用 "
                f"{csi['historical_return_attempts_used']} 次。"
            ),
            f"2. PCF/IOPV（30%）：{pcf['full_quality_days']}/{pcf['quality_day_gate']} 个完整质量日。",
            "",
            "70%/30%仅保留原计划证据，不构成对已终局候选继续投入或执行历史检验的授权。",
            "",
            "## PCF/IOPV 最新直接回执",
            "",
            f"- 回执：`{latest['path']}`",
            f"- 结束时间：`{latest['ended_at']}`",
            f"- 状态：`{latest['collection_status']}`；退出码 `{latest['task_exit_code']}`。",
            f"- 本次是否计质量日：`{str(latest['quality_day_counted']).lower()}`。",
            f"- 失败日志已哈希：`{latest['log_evidence']['sha256']}`。",
            f"- 原始响应存在：`{str(latest['raw_response_present']).lower()}`。",
            "",
            "`ALREADY_ATTEMPTED` 只表示编排去重，不覆盖直接采集回执，也不增加质量日。",
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
    parser = argparse.ArgumentParser(description="生成优先前瞻研究权威状态 V1.7")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    generated_at = datetime.now(ZoneInfo(str(config["timezone"])))
    report = build_report(config, generated_at)
    _atomic_text(_path(config["outputs"]["json"]), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(_path(config["outputs"]["markdown"]), render_markdown(report))
    latest = report["active_research_streams"][1]["latest_task_authority"]
    print(f"权威状态：{report['overall_research_status']}")
    print(f"唯一新冻结测试：{report['only_new_frozen_test']}")
    print(f"PCF/IOPV 最新直接状态：{latest['collection_status']}，退出码 {latest['task_exit_code']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
