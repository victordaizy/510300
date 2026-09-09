"""只读审计桶2严格前向复制草案，不访问未来收益。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "a_share_bucket2_forward_lowvol_replication_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_root(relative: str) -> Path:
    root = ROOT.resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层不是对象：{path}")
    return value


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("复制协议顶层不是对象")
    return value


def validate_contract(contract: dict[str, Any]) -> list[str]:
    """校验复制规则是否仍等于冻结桶2语义。"""

    errors: list[str] = []
    expected = {
        "project_id": "A_SHARE_BUCKET2_FORWARD_LOWVOL_REPLICATION_V1",
        "parent_project_id": "A_SHARE_BUCKET2_TIME_HOLDOUT_LOWVOL_V1",
        "fixed_bucket": 2,
        "rebalance": 60,
        "horizon": 60,
        "factor_count": 1,
        "score": "low_volatility_20_rank",
        "holdings": 1,
        "retention": 1,
        "initial_cash": 20000.0,
        "lot_size": 100,
        "minimum_trade": 5000.0,
        "maximum_positions": 3,
        "cash_rate": 0.015,
        "minimum_amount": 1000000000.0,
        "minimum_price": 10.0,
        "maximum_one_lot": 6500.0,
        "maximum_gap": 0.095,
        "commission_rate": 0.0003,
        "minimum_commission": 5.0,
        "stamp_duty_rate": 0.0005,
        "base_slippage": 5.0,
        "stress_slippage": 15.0,
        "annualized_excess": 0.20,
        "stress_annualized_excess": 0.20,
        "bootstrap_repetitions": 5000,
        "bootstrap_block_length": 20,
        "bootstrap_confidence": 0.95,
        "positive_segments": 2,
        "minimum_days": 1800,
        "minimum_cycles": 30,
        "attempts": 1,
    }
    actual = {
        "project_id": contract["protocol"]["project_id"],
        "parent_project_id": contract["protocol"]["parent_project_id"],
        "fixed_bucket": contract["split"]["fixed_bucket"],
        "rebalance": contract["periods"]["rebalance_every_trading_days"],
        "horizon": contract["periods"]["target_horizon_trading_days"],
        "factor_count": contract["formula"]["factor_count"],
        "score": contract["formula"]["score"],
        "holdings": contract["selection"]["holdings"],
        "retention": contract["selection"]["retention_score_rank"],
        "initial_cash": contract["account"]["initial_cash_cny"],
        "lot_size": contract["account"]["lot_size"],
        "minimum_trade": contract["account"]["minimum_trade_notional_cny"],
        "maximum_positions": contract["account"]["maximum_positions"],
        "cash_rate": contract["account"]["cash_annual_rate"],
        "minimum_amount": contract["universe"]["minimum_20d_average_amount_cny"],
        "minimum_price": contract["universe"]["minimum_signal_price_cny"],
        "maximum_one_lot": contract["universe"]["maximum_signal_price_for_one_lot_cny"],
        "maximum_gap": contract["universe"]["maximum_open_total_return_gap_for_trade"],
        "commission_rate": contract["costs"]["commission_rate"],
        "minimum_commission": contract["costs"]["minimum_commission_cny"],
        "stamp_duty_rate": contract["costs"]["current_stamp_duty_sell_rate"],
        "base_slippage": contract["costs"]["base_slippage_bps_per_leg"],
        "stress_slippage": contract["costs"]["stress_slippage_bps_per_leg"],
        "annualized_excess": contract["evaluation"]["annualized_excess_minimum"],
        "stress_annualized_excess": contract["evaluation"]["stress_annualized_excess_minimum"],
        "bootstrap_repetitions": contract["evaluation"]["bootstrap_repetitions"],
        "bootstrap_block_length": contract["evaluation"]["bootstrap_block_length_trading_days"],
        "bootstrap_confidence": contract["evaluation"]["bootstrap_confidence_level"],
        "positive_segments": contract["evaluation"]["positive_predefined_segments_minimum"],
        "minimum_days": contract["periods"]["minimum_forward_trading_days"],
        "minimum_cycles": contract["periods"]["minimum_closed_holding_cycles"],
        "attempts": contract["blindness"]["unseal_attempts_allowed"],
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            errors.append(f"{field}应为{expected_value!r}，实际为{actual[field]!r}")
    segments = contract["periods"]["evaluation_segments"]
    if [(item["first_cycle"], item["last_cycle"]) for item in segments] != [
        (1, 10),
        (11, 20),
        (21, 30),
    ]:
        errors.append("评价周期段必须固定为1-10、11-20、21-30")
    if contract["universe"]["legacy_all_ten_raw_features_must_be_finite"] is not True:
        errors.append("必须保留旧实现的全部10个原始特征完整性资格条件")
    if contract["formula"]["fit_required"] is not False:
        errors.append("固定低波公式不得拟合")
    if contract["formula"]["parameter_search_allowed"] is not False:
        errors.append("固定低波公式不得搜索参数")
    if contract["evaluation"]["all_gates_required"] is not True:
        errors.append("一次复制必须要求全部门槛通过")
    data_contract = contract["data_contract"]
    if data_contract["total_return_link"] != "previous_raw_close_divided_by_current_pre_close":
        errors.append("总收益链接规则不得变化")
    if data_contract["missing_value_imputation_allowed"] is not False:
        errors.append("未来输入不得插补")
    if data_contract["stale_date_substitution_allowed"] is not False:
        errors.append("未来输入不得用旧日期替代")
    if contract["blindness"]["consume_attempt_before_reading_return_values"] is not True:
        errors.append("必须在读取收益值前消耗唯一机会")
    if contract["blindness"]["technical_failure_after_unseal_consumes_attempt"] is not True:
        errors.append("开封后的技术失败必须消耗唯一机会")
    if contract["blindness"]["receipt_external_worm_copy_required_before_result_publication"] is not True:
        errors.append("结果发布前必须保存外部不可改写收据副本")
    if contract["safety"].get("research_only") is not True:
        errors.append("复制协议必须保持研究层")
    if contract["safety"].get("allow_as_510300_alpha_input") is not False:
        errors.append("桶2复制不得成为510300主系统输入")
    if any(contract["safety"].get(field) is not False for field in (
        "paper_signal_enabled",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    )):
        errors.append("研究安全开关未全部关闭")
    return errors


def audit() -> dict[str, Any]:
    contract = _config()
    errors = validate_contract(contract)
    parent = contract["frozen_parent_evidence"]
    event_path = _inside_root(parent["event"])
    evidence_path = _inside_root(parent["evidence_manifest"])
    event = _json(event_path)
    evidence = _json(evidence_path)
    actual_evidence_hash = sha256(evidence_path)
    if actual_evidence_hash != parent["evidence_manifest_sha256"]:
        errors.append("旧桶2冻结证据清单哈希变化")
    if event.get("event_type") != "NO_FURTHER_BUCKET2_ANALYSIS":
        errors.append("旧桶2永久停止事件缺失")
    if evidence.get("research_status") != parent["required_parent_status"]:
        errors.append("旧桶2冻结状态不一致")
    evidence_mismatches = []
    for item in evidence.get("evidence", []):
        path = _inside_root(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            evidence_mismatches.append(item["path"])
    if evidence_mismatches:
        errors.append(f"旧桶2冻结证据变化：{evidence_mismatches}")

    governance = _json(_inside_root(contract["governance_gate"]["status_source"]))
    governance_status = str(governance.get("overall_status"))
    current_date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    governance_current = str(governance.get("as_of_date")) == current_date
    manifest_exists = _inside_root(contract["paths"]["protocol_manifest"]).exists()
    started_exists = _inside_root(contract["paths"]["unseal_started_receipt"]).exists()
    terminal_exists = _inside_root(contract["paths"]["terminal_receipt"]).exists()
    future_paths = (
        "master_snapshots",
        "daily_checkpoints",
        "benchmark_checkpoints",
        "collection_ledger",
        "sealed_input_manifest",
        "maturity_status",
        "result_json",
        "result_markdown",
    )
    existing_future_paths = [
        contract["paths"][key]
        for key in future_paths
        if _inside_root(contract["paths"][key]).exists()
    ]
    superseded = contract["protocol"].get("status") == "SUPERSEDED_BEFORE_FREEZE_VISIBLE_SHADOW_SELECTED"
    eligible_to_freeze = (
        not errors
        and not superseded
        and governance_status == contract["governance_gate"]["required_overall_status"]
        and governance_current
        and not manifest_exists
        and not existing_future_paths
        and not started_exists
        and not terminal_exists
    )
    if errors:
        status = "BLOCKED_CONTRACT_OR_PARENT_FREEZE_BREACH"
    elif superseded:
        status = "SUPERSEDED_BEFORE_FREEZE_VISIBLE_SHADOW_SELECTED"
    elif governance_status != "PASS" or not governance_current:
        status = "DRAFT_GOVERNANCE_BLOCKED"
    elif manifest_exists:
        status = "PROTOCOL_ALREADY_FROZEN"
    elif existing_future_paths:
        status = "BLOCKED_FUTURE_DATA_EXISTS_BEFORE_FREEZE"
    else:
        status = "ELIGIBLE_TO_FREEZE_PROTOCOL_ONLY"
    return {
        "project_id": contract["protocol"]["project_id"],
        "status": status,
        "eligible_to_freeze": eligible_to_freeze,
        "governance_status": governance_status,
        "governance_as_of_current_date": governance_current,
        "parent_freeze_manifest_sha256": actual_evidence_hash,
        "parent_evidence_mismatches": evidence_mismatches,
        "protocol_manifest_exists": manifest_exists,
        "future_paths_existing_before_freeze": existing_future_paths,
        "unseal_started": started_exists,
        "terminal_exists": terminal_exists,
        "contract_errors": errors,
        "performance_values_read": False,
        "next_action": (
            "DO_NOT_START_STRICT_BLIND_PROTOCOL"
            if superseded
            else "WAIT_FOR_GOVERNANCE_PASS"
            if status == "DRAFT_GOVERNANCE_BLOCKED"
            else "REVIEW_STATUS"
        ),
    }


def main() -> int:
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["contract_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
