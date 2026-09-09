"""V12只增补评价器净敞口字段的模式修正版。"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from research.digital_asset_basis_ranked_spot_long_v12 import (
    build_daily_targets,
    load_contract as load_base_contract,
    load_visible_inputs,
    render_markdown as render_base_markdown,
    run_portfolio_backtest,
)
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONSERVATIVE_EVALUATOR_CONFIG,
    atomic_json,
    atomic_parquet,
    atomic_text,
)
from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    evaluate_historical_returns_zero_variance_v2,
    load_correction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config"
    / "digital_asset_basis_ranked_spot_long_v12r_schema_correction.yaml"
)


def _load_correction(path: Path = CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V12R模式修正配置必须是YAML对象")
    return payload


def _expected_contract(correction: dict[str, Any]) -> dict[str, Any]:
    base_path = ROOT / correction["base"]["config"]
    contract = copy.deepcopy(load_base_contract(base_path))
    contract["protocol"]["candidate_id"] = correction["protocol"]["candidate_id"]
    contract["protocol"]["parent_protocol_id"] = correction["protocol"][
        "parent_protocol_id"
    ]
    contract["protocol"]["status"] = correction["protocol"]["status"]
    contract["protocol"]["correction_of_candidate_id"] = correction["protocol"][
        "correction_of_candidate_id"
    ]
    contract["risk"]["maximum_absolute_net_exposure"] = float(
        correction["correction"]["value"]
    )
    contract["outputs"] = copy.deepcopy(correction["outputs"])
    contract["safety"] = copy.deepcopy(correction["safety"])
    return contract


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    correction = _load_correction(path)
    contract = _expected_contract(correction)
    validate_contract(contract, correction)
    return contract


def validate_contract(
    contract: dict[str, Any],
    correction: dict[str, Any] | None = None,
) -> None:
    correction = _load_correction() if correction is None else correction
    expected = _expected_contract(correction)
    failures: list[str] = []
    if contract != expected:
        failures.append("contract_differs_from_exact_schema_correction")
    if correction.get("correction", {}).get("path") != "risk.maximum_absolute_net_exposure":
        failures.append("correction_path")
    if float(correction.get("correction", {}).get("value", float("nan"))) != 1.0:
        failures.append("correction_value")
    for key in (
        "changes_signal_parameters",
        "changes_portfolio_parameters",
        "changes_costs",
        "changes_sample_or_inputs",
        "changes_trade_engine",
    ):
        if correction.get("correction", {}).get(key) is not False:
            failures.append(key)
    if any(bool(value) for value in contract.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"V12R模式修正被扩大或损坏：{sorted(set(failures))}")


def render_markdown(report: dict[str, Any]) -> str:
    return render_base_markdown(report).replace(
        "数字资产基差排序现货单多V12可见期结果",
        "数字资产基差排序现货单多V12R模式修正可见期结果",
    )


def run_visible(
    contract: dict[str, Any],
    *,
    manifest_verification: dict[str, Any],
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    validate_contract(contract)
    spot, futures, fx, benchmark, input_audit = load_visible_inputs(contract)
    targets, features, signal_audit = build_daily_targets(spot, futures, contract)
    daily, trades, portfolio_audit, context_audit = run_portfolio_backtest(
        targets, spot, fx, benchmark, contract
    )
    evaluator_contract = load_correction_contract(CONSERVATIVE_EVALUATOR_CONFIG)
    evaluation = evaluate_historical_returns_zero_variance_v2(
        daily,
        contract,
        evaluator_contract,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    passed = bool(evaluation["all_visible_gates_pass"])
    report = {
        "schema_version": "1.0.0",
        "report_id": "DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_VISIBLE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": "VISIBLE_ONLY",
        "status": (
            "VISIBLE_PASS_40PCT_HIGH_SHARPE_RESEARCH_SCREEN_ONLY"
            if passed
            else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        ),
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "correction_of_candidate_id": contract["protocol"][
            "correction_of_candidate_id"
        ],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {
            "start": contract["historical_partition"]["visible_start"],
            "end": contract["historical_partition"]["visible_end"],
        },
        "objective": {
            "initial_capital_cny": contract["account"]["initial_capital_cny"],
            "minimum_annualized_net_excess": contract["visible_gates"][
                "minimum_annualized_net_excess"
            ],
            "minimum_strategy_net_sharpe": contract["visible_gates"][
                "minimum_strategy_net_sharpe"
            ],
            "user_transaction_fee_rate_per_leg": contract["account"][
                "user_transaction_fee_rate_per_leg"
            ],
        },
        "schema_correction": _load_correction()["correction"],
        "manifest_verification": manifest_verification,
        "data_audit": {
            "inputs": input_audit,
            "signal": signal_audit,
            "portfolio": portfolio_audit,
            "daily_context": context_audit,
            "selection_feature_rows": int(len(features)),
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "external_forward_validation_required": passed,
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "paper_or_live_authorized": False,
            "next_step": (
                "可见期全部通过；锁定V12R并另设真正前瞻复验，当前不生成任何交易信号"
                if passed
                else "冻结拒绝V12R，不修改V12经济参数、成本、样本、交易引擎或评价门救回"
            ),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    outputs = contract["outputs"]
    atomic_parquet(ROOT / outputs["visible_daily_returns"], daily)
    atomic_parquet(ROOT / outputs["visible_targets"], targets)
    atomic_parquet(ROOT / outputs["visible_trades"], trades)
    atomic_json(ROOT / outputs["input_audit_json"], report["data_audit"])
    atomic_json(ROOT / outputs["visible_report_json"], report)
    atomic_text(ROOT / outputs["visible_report_markdown"], render_markdown(report))
    return report
