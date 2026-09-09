from __future__ import annotations

from research.expectations_news_price_response_chain_pit_preflight_v1 import (
    evaluate_pit_chain_gates,
    preflight_adjudication,
)


def test_any_hard_gate_failure_blocks_price_response_measurement() -> None:
    gates = evaluate_pit_chain_gates(
        {
            "official_pit_weight_row_share": 0.0,
            "periodic_event_archive_pass": True,
            "periodic_exact_timestamp_share": 0.01,
            "official_fact_extraction_success_rate": 0.55,
            "first_public_fact_coverage": 0.52,
            "official_fact_contract_pass": False,
            "complete_event_clustering_universe": False,
            "event_price_clock_admitted": False,
        },
        {
            "minimum_official_pit_weight_row_share": 0.95,
            "minimum_exact_publication_timestamp_share": 0.95,
            "minimum_official_fact_extraction_success_rate": 0.90,
            "minimum_first_public_fact_coverage": 0.90,
        },
    )
    result = preflight_adjudication(gates)

    assert len(gates) == 7
    assert gates.loc[
        gates["gate_id"].eq("G2_OFFICIAL_PERIODIC_EVENT_ARCHIVE_CHRONOLOGY"),
        "passed",
    ].iloc[0]
    assert result["all_required_input_gates_passed"] is False
    assert result["chain_status"] == "NO_VIEW_PIT_CHAIN_INPUT_ADMISSION_FAILED"
    assert result["market_price_value_read_allowed"] is False
    assert result["price_response_measurement_allowed"] is False
    assert result["portfolio_action"] == "ABSTAIN"


def test_all_input_gates_can_admit_measurement_but_never_portfolio() -> None:
    gates = evaluate_pit_chain_gates(
        {
            "official_pit_weight_row_share": 1.0,
            "periodic_event_archive_pass": True,
            "periodic_exact_timestamp_share": 1.0,
            "official_fact_extraction_success_rate": 1.0,
            "first_public_fact_coverage": 1.0,
            "official_fact_contract_pass": True,
            "complete_event_clustering_universe": True,
            "event_price_clock_admitted": True,
        },
        {
            "minimum_official_pit_weight_row_share": 0.95,
            "minimum_exact_publication_timestamp_share": 0.95,
            "minimum_official_fact_extraction_success_rate": 0.90,
            "minimum_first_public_fact_coverage": 0.90,
        },
    )
    result = preflight_adjudication(gates)

    assert result["all_required_input_gates_passed"] is True
    assert result["price_response_measurement_allowed"] is True
    assert result["return_prediction_allowed"] is False
    assert result["portfolio_evaluation_allowed"] is False
