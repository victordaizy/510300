"""预期—消息—价格响应链的 PIT 入场门槛裁决。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_pit_chain_gates(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    """按冻结阈值逐项裁决；任一硬门槛失败即禁止读取价格响应。"""

    gates = [
        {
            "gate_id": "G1_OFFICIAL_PIT_MEMBERSHIP_AND_WEIGHT_VINTAGE",
            "metric_name": "official_pit_weight_row_share",
            "actual_value": float(metrics["official_pit_weight_row_share"]),
            "required_value": float(
                thresholds["minimum_official_pit_weight_row_share"]
            ),
            "passed": float(metrics["official_pit_weight_row_share"])
            >= float(thresholds["minimum_official_pit_weight_row_share"]),
            "failure_status": "BLOCKED_NO_OFFICIAL_HISTORICAL_WEIGHT_VINTAGE",
        },
        {
            "gate_id": "G2_OFFICIAL_PERIODIC_EVENT_ARCHIVE_CHRONOLOGY",
            "metric_name": "periodic_event_archive_pass",
            "actual_value": 1.0 if metrics["periodic_event_archive_pass"] else 0.0,
            "required_value": 1.0,
            "passed": bool(metrics["periodic_event_archive_pass"]),
            "failure_status": "BLOCKED_PERIODIC_EVENT_ARCHIVE",
        },
        {
            "gate_id": "G3_EXACT_FIRST_PUBLICATION_TIMESTAMP",
            "metric_name": "periodic_exact_timestamp_share",
            "actual_value": float(metrics["periodic_exact_timestamp_share"]),
            "required_value": float(
                thresholds["minimum_exact_publication_timestamp_share"]
            ),
            "passed": float(metrics["periodic_exact_timestamp_share"])
            >= float(thresholds["minimum_exact_publication_timestamp_share"]),
            "failure_status": "BLOCKED_PERIODIC_PUBLICATION_TIME_DATE_ONLY",
        },
        {
            "gate_id": "G4_OFFICIAL_CORE_FACT_EXTRACTION",
            "metric_name": "official_fact_extraction_success_rate",
            "actual_value": float(metrics["official_fact_extraction_success_rate"]),
            "required_value": float(
                thresholds["minimum_official_fact_extraction_success_rate"]
            ),
            "passed": float(metrics["official_fact_extraction_success_rate"])
            >= float(thresholds["minimum_official_fact_extraction_success_rate"]),
            "failure_status": "BLOCKED_OFFICIAL_FACT_EXTRACTION_COVERAGE",
        },
        {
            "gate_id": "G5_FIRST_PUBLIC_FACT_AND_NO_REVISED_VALUE_AS_PIT",
            "metric_name": "first_public_fact_coverage",
            "actual_value": float(metrics["first_public_fact_coverage"]),
            "required_value": float(thresholds["minimum_first_public_fact_coverage"]),
            "passed": (
                float(metrics["first_public_fact_coverage"])
                >= float(thresholds["minimum_first_public_fact_coverage"])
                and bool(metrics["official_fact_contract_pass"])
            ),
            "failure_status": "BLOCKED_FIRST_PUBLIC_FACT_OR_REVISION_CONTROL",
        },
        {
            "gate_id": "G6_COMPLETE_EVENT_CLUSTERING_UNIVERSE",
            "metric_name": "complete_event_clustering_universe",
            "actual_value": 1.0
            if metrics["complete_event_clustering_universe"]
            else 0.0,
            "required_value": 1.0,
            "passed": bool(metrics["complete_event_clustering_universe"]),
            "failure_status": "BLOCKED_INCOMPLETE_CLUSTERING_UNIVERSE",
        },
        {
            "gate_id": "G7_CLOCK_ALIGNED_EVENT_PRICE_RESPONSE",
            "metric_name": "event_price_clock_admitted",
            "actual_value": 1.0 if metrics["event_price_clock_admitted"] else 0.0,
            "required_value": 1.0,
            "passed": bool(metrics["event_price_clock_admitted"]),
            "failure_status": "BLOCKED_EVENT_PRICE_CLOCK",
        },
    ]
    result = pd.DataFrame(gates)
    result["gate_status"] = result.apply(
        lambda row: "PASS" if row["passed"] else row["failure_status"], axis=1
    )
    result["market_price_value_read_allowed_after_gate"] = result["passed"].cumprod().astype(bool)
    result["return_prediction_allowed"] = False
    result["portfolio_evaluation_allowed"] = False
    return result


def preflight_adjudication(gates: pd.DataFrame) -> dict[str, Any]:
    """汇总入场结果并保持 NO_VIEW/ABSTAIN 边界。"""

    all_pass = bool(gates["passed"].all())
    failed = gates.loc[~gates["passed"], ["gate_id", "gate_status"]]
    return {
        "all_required_input_gates_passed": all_pass,
        "passed_gate_count": int(gates["passed"].sum()),
        "total_gate_count": int(len(gates)),
        "failed_gates": failed.to_dict(orient="records"),
        "chain_status": (
            "PIT_CHAIN_INPUT_ADMITTED_PRICE_RESPONSE_MEASUREMENT_MAY_START"
            if all_pass
            else "NO_VIEW_PIT_CHAIN_INPUT_ADMISSION_FAILED"
        ),
        "market_price_value_read_allowed": all_pass,
        "price_response_measurement_allowed": all_pass,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "portfolio_action": "ABSTAIN",
    }
