"""一次性运行并核验 510300_ASYMMETRIC_STRESS_HAZARD_V1 的 BAD10 普查。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_v1 import (
    adjudicate_g1,
    build_bad10_origin_panel,
    build_census_summary,
    merge_bad10_events,
    prepare_cash_dividends,
    prepare_etf_prices,
    select_independent_non_event_blocks,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    atomic_write_text_new,
    canonical_sha256,
    create_exclusive_claim,
    ensure_paths_committed_at_head,
    file_evidence,
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    strict_json_text,
    validate_authority_state,
    validate_registered_inputs,
    verify_manifest_files,
)


DEFAULT_CONFIG = ROOT / "config/510300_asymmetric_stress_hazard_v1.yaml"


class OneShotConsumedError(EvidenceContractError):
    """一次性普查已经拥有 claim 或结果时拒绝再次运行。"""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("协议配置必须是对象")
    return payload


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(name): _project_path(str(relative))
        for name, relative in config["artifacts"].items()
    }


def _registered_record_without_resolved_path(record: Any) -> dict[str, Any]:
    payload = record.to_dict()
    payload.pop("resolved_path", None)
    return payload


def verify_frozen_contract(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """只校验字节、Git 与数据根；不读取价格列或标签。"""

    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    artifacts = _artifact_paths(config)
    manifest = read_json_strict(artifacts["protocol_manifest"])
    if not isinstance(manifest, dict):
        raise EvidenceContractError("冻结 manifest 必须是对象")
    if manifest.get("status") != "FROZEN_BEFORE_BAD10_LABEL_CENSUS":
        raise EvidenceContractError("manifest 状态不允许 BAD10 普查")
    verify_manifest_files(manifest, project_root=ROOT)
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise EvidenceContractError("协议配置字节哈希与 manifest 不一致")
    if canonical_sha256(config) != manifest.get("config_canonical_sha256"):
        raise EvidenceContractError("协议配置语义哈希与 manifest 不一致")

    authority_path = _project_path(str(config["authority_contract"]["path"]))
    authority = read_json_strict(authority_path)
    if not isinstance(authority, dict):
        raise EvidenceContractError("权威状态必须是对象")
    validate_authority_state(authority)
    if sha256_file(authority_path) != manifest["authority"]["sha256"]:
        raise EvidenceContractError("权威状态字节与冻结 manifest 不一致")

    registered = validate_registered_inputs(config, project_root=ROOT)
    manifest_inputs = manifest.get("registered_inputs")
    if not isinstance(manifest_inputs, dict):
        raise EvidenceContractError("manifest 缺少 registered_inputs")
    for input_id, evidence in registered.items():
        actual = _registered_record_without_resolved_path(evidence)
        if manifest_inputs.get(input_id) != actual:
            raise EvidenceContractError(f"{input_id} 输入身份与 manifest 不一致")

    required_paths = [
        str(value)
        for value in config["freeze_contract"]["required_committed_before_census"]
    ]
    committed = ensure_paths_committed_at_head(ROOT, required_paths)
    return config, manifest, {
        "authority": authority,
        "registered_inputs": registered,
        "committed_paths": committed,
        "git_head": git_head(ROOT),
    }


def _assert_unconsumed(artifacts: Mapping[str, Path]) -> None:
    existing = {
        name: str(path)
        for name, path in artifacts.items()
        if name != "protocol_manifest" and path.exists()
    }
    if existing:
        if "one_shot_claim" in existing:
            raise OneShotConsumedError(
                f"BAD10 一次性普查已被 claim 消耗：{existing['one_shot_claim']}"
            )
        raise EvidenceContractError(f"普查输出已存在，禁止覆盖：{existing}")


def _atomic_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        if path.exists():
            raise EvidenceContractError(f"并发输出冲突，禁止覆盖：{path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceContractError(f"输出已存在，禁止覆盖：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
            lineterminator="\n",
        )
        if path.exists():
            raise EvidenceContractError(f"并发输出冲突，禁止覆盖：{path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_report(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> str:
    gate = summary["g1"]
    disposition = (
        "G1 通过只允许进入点时来源合同与机制构造阶段；尚未构造任何因子或模型。"
        if gate["passed"]
        else "G1 失败，V1 永久冻结；不得修改阈值、期限、非事件锚点或代理救援。"
    )
    return "\n".join(
        [
            "# 510300_ASYMMETRIC_STRESS_HAZARD_V1 BAD10 独立事件普查",
            "",
            "## 裁决",
            "",
            f"- 状态：`{gate['status']}`",
            f"- 独立 BAD10 压力事件：`{gate['independent_event_count']}` / 要求 `{gate['minimum_independent_events']}`。",
            f"- 独立非事件十日块：`{gate['independent_non_event_block_count']}` / 要求 `{gate['minimum_independent_non_event_blocks']}`。",
            f"- 合格标签原点：`{summary['eligible_origin_count']}`；其中 BAD10 原点 `{summary['bad10_origin_count']}`。",
            f"- 标签固定为未来 `{summary['horizon_market_days']}` 个交易日、路径损失小于或等于 `{summary['loss_threshold']:.2%}`。",
            f"- 冻结 manifest：`{manifest['manifest_payload_sha256']}`。",
            "",
            disposition,
            "",
            "## 研究与交易边界",
            "",
            f"- `RESEARCH_STATE={authority['research_state']}`",
            f"- `MODEL_POSITION_TARGET={authority['model_position_target']}`",
            f"- `ORDER_AUTHORIZATION={authority['order_authorization']}`",
            f"- `ACTUAL_HOLDINGS_STATE={authority['actual_holdings_state']}`",
            f"- `POSITION_IMPACT={authority['position_impact']}`",
            "- 本次没有构造 M、F、T，没有训练模型，没有选择概率阈值，没有读取组合收益、净夏普或回撤。",
            "- `ABSTAIN` 不表示现金目标，也不覆盖用户实际持仓。",
            "",
            "## 标签口径",
            "",
            "信号信息截止于 t 日收盘；以 t+1 日 510300 实际开盘价为入场成本；观察未来十个交易日的实际收盘财富。财富包含持有期间取得资格并自除息日起确认的现金分红应收款。不使用 H00300、同日收盘成交或未来最低价。",
            "",
            "BAD10 原点按入场日至十日终点的闭区间做传递重叠合并。非事件块按冻结的最早合格原点贪心向前选择，彼此及其与压力事件均不重叠。",
            "",
        ]
    )


def _result_payload(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    summary: Mapping[str, Any],
    completed_at: str,
    claim_path: Path,
) -> dict[str, Any]:
    gate = summary["g1"]
    return {
        "program_id": config["program"]["program_id"],
        "version": config["program"]["version"],
        "status": gate["status"],
        "completed_at": completed_at,
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "one_shot_claim": {
            "path": claim_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(claim_path),
        },
        "census": summary,
        "authority": {
            "research_state": authority["research_state"],
            "model_position_target": authority["model_position_target"],
            "order_authorization": authority["order_authorization"],
            "actual_holdings_state": authority["actual_holdings_state"],
            "position_impact": authority["position_impact"],
        },
        "feature_values_constructed": False,
        "model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "paper_shadow_generated": False,
        "order_generated": False,
        "broker_action_performed": False,
        "live_trading_authorized": False,
        "rescue_allowed": False,
        "next_allowed_action": (
            "FREEZE_AND_ADMIT_POINT_IN_TIME_M_F_T_SOURCE_CONTRACTS"
            if gate["passed"]
            else "NONE_V1_REJECTED_FROZEN_NO_RESCUE"
        ),
    }


def _status_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "program_id": result["program_id"],
        "version": result["version"],
        "status": result["status"],
        "completed_at": result["completed_at"],
        "protocol_manifest_payload_sha256": result[
            "protocol_manifest_payload_sha256"
        ],
        "g1": result["census"]["g1"],
        "authority": result["authority"],
        "feature_values_constructed": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "order_generated": False,
        "live_trading_authorized": False,
        "rescue_allowed": False,
        "next_allowed_action": result["next_allowed_action"],
    }


def _write_failure_receipt(
    *,
    artifacts: Mapping[str, Path],
    manifest: Mapping[str, Any] | None,
    exc: Exception,
) -> None:
    failure_path = artifacts["failure_receipt"]
    if failure_path.exists():
        return
    claim_path = artifacts["one_shot_claim"]
    payload = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "status": "PROGRAM_FAILED_AFTER_ONE_SHOT_CLAIM_NO_RERUN",
        "failed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "protocol_manifest_payload_sha256": (
            manifest.get("manifest_payload_sha256") if manifest else None
        ),
        "claim": (
            file_evidence(claim_path, project_root=ROOT)
            if claim_path.is_file()
            else None
        ),
        "position_impact": 0,
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "rerun_allowed": False,
    }
    atomic_write_json_new(failure_path, payload)


def run_census(config_path: Path) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    _assert_unconsumed(artifacts)
    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    claim_payload = {
        "claim_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1_BAD10_CENSUS_ONE_SHOT",
        "claimed_at": created_at,
        "git_head": verification["git_head"],
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "research_state": "DISCOVERY_ONLY",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
        "position_impact": 0,
        "claim_consumption_rule": "ONE_ATTEMPT_ONLY_NO_OVERWRITE_NO_RERUN",
    }
    create_exclusive_claim(artifacts["one_shot_claim"], claim_payload)

    try:
        price_evidence = verification["registered_inputs"]["etf_price"]
        dividend_evidence = verification["registered_inputs"]["etf_dividends"]
        price_contract = config["source_contract"]["etf_price"]
        dividend_contract = config["source_contract"]["etf_dividends"]
        prices_raw = pd.read_parquet(
            Path(price_evidence.resolved_path),
            columns=list(price_contract["required_columns"]),
        )
        dividends_raw = pd.read_csv(
            Path(dividend_evidence.resolved_path),
            usecols=list(dividend_contract["required_columns"]),
        )
        prices = prepare_etf_prices(
            prices_raw,
            expected_symbol=str(price_contract["expected_symbol"]),
            start_date=str(config["program"]["observation_start"]),
            observation_cutoff=str(config["program"]["observation_cutoff"]),
        )
        dividends = prepare_cash_dividends(
            dividends_raw,
            expected_symbol=str(dividend_contract["expected_symbol"]),
        )
        label = config["bad10_label"]
        origin_panel = build_bad10_origin_panel(
            prices,
            dividends,
            horizon_market_days=int(label["horizon_market_days"]),
            loss_threshold=float(label["loss_threshold"]),
        )
        events = merge_bad10_events(origin_panel)
        non_event_blocks = select_independent_non_event_blocks(origin_panel, events)
        gate_contract = config["gates"]["G1_LABEL_IDENTIFIABILITY"]
        gate = adjudicate_g1(
            independent_event_count=len(events),
            independent_non_event_block_count=len(non_event_blocks),
            minimum_independent_events=int(
                gate_contract["minimum_independent_bad10_events"]
            ),
            minimum_independent_non_event_blocks=int(
                gate_contract["minimum_independent_non_event_10d_blocks"]
            ),
        )
        summary = build_census_summary(
            origin_panel,
            events,
            non_event_blocks,
            horizon_market_days=int(label["horizon_market_days"]),
            loss_threshold=float(label["loss_threshold"]),
            gate=gate,
        )
        completed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        result = _result_payload(
            config=config,
            manifest=manifest,
            authority=verification["authority"],
            summary=summary,
            completed_at=completed_at,
            claim_path=artifacts["one_shot_claim"],
        )
        status = _status_payload(result)
        report = _build_report(
            manifest=manifest,
            summary=summary,
            authority=verification["authority"],
        )

        _atomic_parquet_new(artifacts["origin_panel"], origin_panel)
        _atomic_csv_new(artifacts["independent_events"], events)
        _atomic_csv_new(artifacts["independent_non_event_blocks"], non_event_blocks)
        atomic_write_json_new(artifacts["census_result"], result)
        atomic_write_json_new(artifacts["authoritative_status"], status)
        atomic_write_text_new(artifacts["human_report"], report)

        output_names = [
            "origin_panel",
            "independent_events",
            "independent_non_event_blocks",
            "census_result",
            "authoritative_status",
            "human_report",
        ]
        receipt = {
            "program_id": config["program"]["program_id"],
            "version": config["program"]["version"],
            "status": "IMMUTABLE_BAD10_CENSUS_RECEIPT_COMPLETE",
            "completed_at": completed_at,
            "git_head": verification["git_head"],
            "protocol_manifest_payload_sha256": manifest[
                "manifest_payload_sha256"
            ],
            "one_shot_claim": file_evidence(
                artifacts["one_shot_claim"], project_root=ROOT
            ),
            "registered_inputs": {
                input_id: _registered_record_without_resolved_path(evidence)
                for input_id, evidence in verification["registered_inputs"].items()
            },
            "outputs": {
                name: file_evidence(artifacts[name], project_root=ROOT)
                for name in output_names
            },
            "g1_status": gate["status"],
            "g1_passed": gate["passed"],
            "label_changed_after_freeze": False,
            "feature_values_constructed": False,
            "model_trained": False,
            "portfolio_metrics_read": False,
            "position_impact": 0,
            "order_authorization": "NOT_AUTHORIZED",
        }
        receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
        atomic_write_json_new(artifacts["immutable_receipt"], receipt)
        print(strict_json_text(status, pretty=True))
        return result
    except Exception as exc:
        _write_failure_receipt(artifacts=artifacts, manifest=manifest, exc=exc)
        raise


def verify_results(config_path: Path) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    receipt = read_json_strict(artifacts["immutable_receipt"])
    if not isinstance(receipt, dict):
        raise EvidenceContractError("普查回执必须是对象")
    expected_receipt_hash = receipt.get("receipt_payload_sha256")
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_payload_sha256", None)
    if canonical_sha256(receipt_payload) != expected_receipt_hash:
        raise EvidenceContractError("普查回执自身哈希失败")
    if receipt.get("protocol_manifest_payload_sha256") != manifest.get(
        "manifest_payload_sha256"
    ):
        raise EvidenceContractError("普查回执与冻结 manifest 不一致")
    for name, evidence in receipt["outputs"].items():
        path = _project_path(str(evidence["path"]))
        if path != artifacts[name]:
            raise EvidenceContractError(f"回执输出路径不一致：{name}")
        if path.stat().st_size != int(evidence["bytes"]):
            raise EvidenceContractError(f"回执输出字节数不一致：{name}")
        if sha256_file(path) != evidence["sha256"]:
            raise EvidenceContractError(f"回执输出哈希不一致：{name}")
    result = read_json_strict(artifacts["census_result"])
    if not isinstance(result, dict):
        raise EvidenceContractError("普查结果必须是对象")
    if result.get("portfolio_metrics_read") is not False:
        raise EvidenceContractError("普查结果越界读取了组合指标")
    if result.get("feature_values_constructed") is not False:
        raise EvidenceContractError("普查结果越界构造了特征")
    if result.get("model_trained") is not False:
        raise EvidenceContractError("普查结果越界训练了模型")
    validate_authority_state(verification["authority"])
    return {
        "status": "PASS_BAD10_CENSUS_RESULTS_AND_RECEIPT_VERIFIED",
        "git_head": verification["git_head"],
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "receipt_payload_sha256": expected_receipt_hash,
        "g1_status": result["census"]["g1"]["status"],
        "g1_passed": result["census"]["g1"]["passed"],
        "position_impact": 0,
        "order_authorization": "NOT_AUTHORIZED",
    }


def verify_g0(config_path: Path) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    _assert_unconsumed(artifacts)
    return {
        "status": "PASS_G0_READY_FOR_ONE_SHOT_BAD10_CENSUS",
        "git_head": verification["git_head"],
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "committed_path_count": len(verification["committed_paths"]),
        "registered_input_count": len(verification["registered_inputs"]),
        "label_values_read": False,
        "feature_values_constructed": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "order_authorization": "NOT_AUTHORIZED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 510300 非对称压力风险 V1 的一次性 BAD10 普查"
    )
    parser.add_argument(
        "--phase", choices=["g0", "census", "verify-results"], required=True
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if args.phase == "g0":
            print(strict_json_text(verify_g0(args.config), pretty=True))
        elif args.phase == "census":
            run_census(args.config)
        else:
            print(strict_json_text(verify_results(args.config), pretty=True))
        return 0
    except OneShotConsumedError as exc:
        print(f"一次性普查已消耗，拒绝重跑：{exc}")
        return 4
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"程序或契约失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
