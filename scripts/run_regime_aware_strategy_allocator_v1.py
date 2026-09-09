"""运行一次 V1 点时资格与研究风险配置，并写入不可变回执。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.regime_aware_strategy_allocator_v1 import (
    CONFIG,
    ContractError,
    EvidenceError,
    audit_component_statuses,
    evaluate_allocator,
    load_contract,
    load_jsonl,
    parse_zoned_datetime,
    render_markdown,
    sha256_file,
)
from scripts.freeze_regime_aware_strategy_allocator_v1 import MANIFEST, verify


FAILURE_STATUS_JSON = ROOT / "reports" / "research" / "regime_aware_strategy_allocator_v1_status.json"
FAILURE_STATUS_MARKDOWN = ROOT / "reports" / "research" / "REGIME_AWARE_STRATEGY_ALLOCATOR_V1_STATUS.md"
FAILURE_RECEIPT_DIRECTORY = ROOT / "reports" / "audit" / "regime_aware_strategy_allocator_v1_runs"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING_NO_VIEW", "path": path.relative_to(ROOT).as_posix(), "bytes": None, "sha256": None}
    if not path.is_file():
        raise EvidenceError(f"输入路径不是文件：{path}")
    return {
        "status": "PRESENT",
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_previous_receipt_weights(
    path: Path | None,
    *,
    contract: dict[str, Any],
    manifest_verification: dict[str, Any],
    decision_at: datetime,
) -> dict[str, float] | None:
    if path is None:
        return None
    resolved = path.resolve()
    receipt_directory = (
        ROOT / PurePosixPath(contract["outputs"]["immutable_receipt_directory"])
    ).resolve()
    if resolved.parent != receipt_directory:
        raise EvidenceError("前次权重只能来自冻结不可变回执目录")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError("前次回执必须是 JSON 对象")
    if payload.get("immutable_receipt") is not True:
        raise EvidenceError("前次回执未声明 immutable_receipt=true")
    if payload.get("receipt_id") != resolved.stem:
        raise EvidenceError("前次回执文件名与 receipt_id 不一致")
    if payload.get("candidate_id") != contract["protocol"]["candidate_id"]:
        raise EvidenceError("前次回执 candidate_id 不一致")
    if payload.get("run_status") != "SUCCESS":
        raise EvidenceError("失败回执不得作为前次风险权重")
    if payload.get("manifest_sha256") != manifest_verification["manifest_sha256"]:
        raise EvidenceError("前次回执未绑定当前冻结清单")
    prior_decision = parse_zoned_datetime(payload.get("decision_at"), "前次回执.decision_at")
    prior_generated = parse_zoned_datetime(payload.get("generated_at"), "前次回执.generated_at")
    if not prior_decision < decision_at:
        raise EvidenceError("前次回执决策时点必须严格早于当前决策")
    if prior_generated > decision_at:
        raise EvidenceError("当前决策时点尚不可得前次回执")
    safety = payload.get("safety")
    if not isinstance(safety, dict) or not safety or any(value is not False for value in safety.values()):
        raise EvidenceError("前次回执安全边界不完整或存在启用项")
    weights = payload.get("research_risk_weights")
    if not isinstance(weights, dict):
        raise EvidenceError("前次回执缺少 research_risk_weights")
    cash_asset_id = contract["allocation"]["cash_asset_id"]
    if payload.get("cash_weight") != weights.get(cash_asset_id):
        raise EvidenceError("前次回执现金权重与完整权重映射不一致")
    return weights


def _receipt_id(generated_at: datetime, manifest_sha256: str) -> str:
    utc = generated_at.astimezone(timezone.utc)
    return f"{utc:%Y%m%dT%H%M%S%fZ}_{os.getpid()}_{manifest_sha256[:12]}"


def _write_failure_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """冻结系统运行失败时仍保留不可变回执；缺少清单时不调用。"""

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    manifest_sha = sha256_file(MANIFEST)
    receipt_id = _receipt_id(generated_at, manifest_sha)
    receipt_path = FAILURE_RECEIPT_DIRECTORY / f"{receipt_id}.json"
    if receipt_path.exists():
        raise FileExistsError(f"失败回执已存在：{receipt_path}")
    payload.update(
        {
            "generated_at": generated_at.isoformat(),
            "receipt_id": receipt_id,
            "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": manifest_sha,
        }
    )
    markdown = "\n".join(
        [
            "# 市场状态感知策略资格与风险配置器 V1 失败状态",
            "",
            f"- 生成时间：`{payload['generated_at']}`",
            f"- 主状态：`{payload['status']}`",
            f"- 错误类型：`{payload['error_type']}`",
            f"- 错误：{payload['error']}",
            "- 策略资格声明：否",
            "- 现金权重声明：无；输入或协议失败不能伪造成配置结果。",
            "- Paper、Shadow、仓位、订单、账户连接和实盘：全部关闭。",
            "",
        ]
    )
    atomic_json(FAILURE_STATUS_JSON, payload)
    atomic_text(FAILURE_STATUS_MARKDOWN, markdown)
    receipt = {
        "schema_version": "1.0.0",
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "generated_at": payload["generated_at"],
        "candidate_id": payload["candidate_id"],
        "run_status": "FAILED",
        "research_status": "INPUT_OR_PROTOCOL_FAILURE",
        "error_type": payload["error_type"],
        "error": payload["error"],
        "manifest_path": payload["manifest_path"],
        "manifest_sha256": manifest_sha,
        "status_json": {
            "path": FAILURE_STATUS_JSON.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(FAILURE_STATUS_JSON),
        },
        "status_markdown": {
            "path": FAILURE_STATUS_MARKDOWN.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(FAILURE_STATUS_MARKDOWN),
        },
        "strategy_eligibility_claimed": False,
        "cash_weight_claimed": None,
        "winner_chasing_input_used": False,
        "live_trading_authorized": False,
    }
    atomic_json(receipt_path, receipt)
    return {
        "status_json": FAILURE_STATUS_JSON.relative_to(ROOT).as_posix(),
        "status_markdown": FAILURE_STATUS_MARKDOWN.relative_to(ROOT).as_posix(),
        "receipt": receipt_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _write_success_artifacts(
    report: dict[str, Any],
    contract: dict[str, Any],
    *,
    no_write: bool,
) -> dict[str, Any]:
    outputs = contract["outputs"]
    status_json = ROOT / PurePosixPath(outputs["latest_status_json"])
    status_markdown = ROOT / PurePosixPath(outputs["latest_status_markdown"])
    receipt_path = ROOT / PurePosixPath(report["receipt_path"])
    markdown = render_markdown(report)
    if no_write:
        return {
            "status_json": status_json.relative_to(ROOT).as_posix(),
            "status_markdown": status_markdown.relative_to(ROOT).as_posix(),
            "receipt": receipt_path.relative_to(ROOT).as_posix(),
            "write_skipped": True,
        }
    if receipt_path.exists():
        raise FileExistsError(f"不可变回执已存在：{receipt_path}")
    atomic_json(status_json, report)
    atomic_text(status_markdown, markdown)
    receipt = {
        "schema_version": "1.0.0",
        "receipt_id": report["receipt_id"],
        "immutable_receipt": True,
        "generated_at": report["generated_at"],
        "decision_at": report["decision_at"],
        "candidate_id": report["candidate_id"],
        "run_status": "SUCCESS",
        "research_status": report["status"],
        "view_status": report["view_status"],
        "eligible_strategy_count": report["eligible_strategy_count"],
        "cash_weight": report["allocation"]["cash_weight"],
        "research_risk_weights": report["allocation"]["research_risk_weights"],
        "manifest_path": report["manifest_verification"]["manifest_path"],
        "manifest_sha256": report["manifest_verification"]["manifest_sha256"],
        "input_snapshots": report["input_snapshots"],
        "status_json": {
            "path": status_json.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(status_json),
        },
        "status_markdown": {
            "path": status_markdown.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(status_markdown),
        },
        "report_payload_sha256": canonical_sha256(report),
        "winner_chasing_input_used": False,
        "historical_return_ranking_used": False,
        "cash_is_valid_state": True,
        "safety": report["safety"],
    }
    atomic_json(receipt_path, receipt)
    return {
        "status_json": status_json.relative_to(ROOT).as_posix(),
        "status_json_sha256": sha256_file(status_json),
        "status_markdown": status_markdown.relative_to(ROOT).as_posix(),
        "status_markdown_sha256": sha256_file(status_markdown),
        "receipt": receipt_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
        "write_skipped": False,
    }


def run(
    *,
    decision_at: datetime,
    previous_receipt: Path | None = None,
    no_write: bool = False,
) -> dict[str, Any]:
    runtime_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if decision_at > runtime_now:
        raise EvidenceError("配置决策时点不得晚于当前可得时间")
    verification = verify()
    if verification["failure_count"]:
        raise ContractError(f"V1 清单验证失败：{verification['failures']}")
    contract = load_contract(CONFIG)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    environment_path = ROOT / PurePosixPath(contract["inputs"]["environment_evidence_jsonl"])
    strategy_path = ROOT / PurePosixPath(contract["inputs"]["strategy_evidence_jsonl"])
    environment_records = load_jsonl(environment_path, allow_missing=True)
    strategy_records = load_jsonl(strategy_path, allow_missing=True)
    previous_weights = load_previous_receipt_weights(
        previous_receipt,
        contract=contract,
        manifest_verification=verification,
        decision_at=decision_at,
    )
    report = evaluate_allocator(
        contract,
        environment_records,
        strategy_records,
        manifest,
        decision_at=decision_at,
        previous_weights=previous_weights,
        root=ROOT,
    )
    report["component_statuses"] = audit_component_statuses(
        contract,
        manifest["component_dependencies"],
        root=ROOT,
    )
    report["manifest_verification"] = verification
    report["input_snapshots"] = {
        "environment_evidence": input_snapshot(environment_path),
        "strategy_evidence": input_snapshot(strategy_path),
        "previous_receipt": (
            {"status": "NOT_PROVIDED", "path": None, "bytes": None, "sha256": None}
            if previous_receipt is None
            else input_snapshot(previous_receipt.resolve())
        ),
        "forward_observation_ledger": input_snapshot(
            ROOT / PurePosixPath(contract["inputs"]["forward_observation_ledger_jsonl"])
        ),
    }
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    report["generated_at"] = generated_at.isoformat()
    report["receipt_id"] = _receipt_id(generated_at, verification["manifest_sha256"])
    report["receipt_path"] = (
        PurePosixPath(contract["outputs"]["immutable_receipt_directory"])
        / f"{report['receipt_id']}.json"
    ).as_posix()
    report["artifact_paths"] = {
        "latest_status_json": contract["outputs"]["latest_status_json"],
        "latest_status_markdown": contract["outputs"]["latest_status_markdown"],
        "immutable_receipt": report["receipt_path"],
    }
    report["artifact_write"] = _write_success_artifacts(report, contract, no_write=no_write)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 V1 点时策略资格与研究风险配置")
    parser.add_argument(
        "--decision-at",
        help="带时区 ISO 决策时点；省略时使用当前亚洲/上海时间",
    )
    parser.add_argument(
        "--previous-receipt",
        type=Path,
        help="可选的前次不可变成功回执；只用于风险增加上限和成本估算",
    )
    parser.add_argument("--no-write", action="store_true", help="只验证和打印，不写状态或回执")
    args = parser.parse_args()
    try:
        decision_at = (
            datetime.now(ZoneInfo("Asia/Shanghai"))
            if args.decision_at is None
            else parse_zoned_datetime(args.decision_at, "--decision-at")
        )
        payload = run(
            decision_at=decision_at,
            previous_receipt=args.previous_receipt,
            no_write=args.no_write,
        )
    except (ContractError, EvidenceError, FileNotFoundError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": "1.0.0",
            "candidate_id": "REGIME_AWARE_STRATEGY_ALLOCATOR_V1",
            "run_status": "FAILED",
            "status": "INPUT_OR_PROTOCOL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cash_weight_claimed": None,
            "strategy_eligibility_claimed": False,
            "paper_position_generation": False,
            "shadow_signal_generation": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_or_exchange_connection_enabled": False,
            "live_trading_authorized": False,
        }
        if MANIFEST.is_file() and not args.no_write:
            try:
                payload["artifact_write"] = _write_failure_artifacts(payload)
            except (OSError, ValueError, FileExistsError) as receipt_exc:
                payload["failure_receipt_error"] = str(receipt_exc)
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
