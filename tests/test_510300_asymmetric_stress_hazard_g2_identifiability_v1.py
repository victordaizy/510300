from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.asymmetric_stress_hazard_g2_identifiability_v1 import (
    FAIL_STATUS,
    PASS_STATUS,
    adjudicate_temporal_identifiability,
    build_daily_necessary_coverage,
    build_potential_origin_ceiling,
    build_preflight_result,
    load_config,
    parse_fixed_subperiods,
    prepare_constituent_presence,
    prepare_industry_presence,
    prepare_membership_presence,
    summarize_subperiod_coverage,
    validate_protocol,
)
from research.project_evidence_contract_v1 import EvidenceContractError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_asymmetric_stress_hazard_v1_g2_identifiability.yaml"
MANIFEST = (
    ROOT
    / "config/510300_asymmetric_stress_hazard_v1_g2_identifiability_manifest.json"
)
RECEIPT = ROOT / "reports/audit/510300_stress_hazard_v1_g2_identifiability_receipt.json"


def _synthetic_inputs(
    *, count: int = 40, members: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=count)
    symbols = [f"{index:06d}.SZ" for index in range(1, members + 1)]
    membership = pd.MultiIndex.from_product(
        [dates, symbols], names=["membership_date", "symbol"]
    ).to_frame(index=False)
    membership["index_code"] = "000300"
    constituent = membership.rename(
        columns={"membership_date": "date", "symbol": "con_code"}
    ).copy()
    constituent["total_return_close"] = 100.0
    industry = membership[["membership_date", "symbol"]].copy()
    industry["industry_l1_code"] = np.resize(
        np.array(["801010", "801020", "801030"], dtype=object), len(industry)
    )
    industry["mapping_status"] = "PIT_AVAILABLE_BY_MARKET_CLOSE"
    return membership, constituent, industry


def _prepared_daily(count: int = 40) -> pd.DataFrame:
    membership_raw, constituent_raw, industry_raw = _synthetic_inputs(count=count)
    membership = prepare_membership_presence(
        membership_raw,
        expected_index_code="000300",
        start=pd.Timestamp("2024-01-02"),
        cutoff=pd.Timestamp("2026-08-14"),
    )
    constituent = prepare_constituent_presence(constituent_raw)
    industry = prepare_industry_presence(
        industry_raw,
        required_mapping_status="PIT_AVAILABLE_BY_MARKET_CLOSE",
    )
    return build_daily_necessary_coverage(
        membership,
        constituent,
        industry,
        expected_members_per_session=3,
    )


def test_protocol_freezes_four_periods_and_no_result_access() -> None:
    config = load_config(CONFIG)
    validate_protocol(config)
    periods = parse_fixed_subperiods(config)
    assert [item.period_id for item in periods] == [
        "P1_2015_2017",
        "P2_2018_2020",
        "P3_2021_2023",
        "P4_2024_CUTOFF",
    ]
    assert config["program"]["feature_values_may_be_constructed_in_this_stage"] is False
    assert config["program"]["bad10_label_artifacts_may_be_read_in_this_stage"] is False
    assert config["program"]["portfolio_evaluation_allowed"] is False
    assert config["temporal_identifiability_preflight"][
        "repartition_after_result_allowed"
    ] is False


def test_feature_micro_semantics_are_frozen_without_search_freedom() -> None:
    config = load_config(CONFIG)
    feature = config["frozen_feature_construction"]
    assert feature["causal_percentile"]["current_observation_excluded"] is True
    assert feature["all_three_channels_required_for_each_composite"] is True
    assert feature["F"]["F1"]["valid_member_count_required"] == 300
    assert feature["F"]["F3"]["window"].startswith("20_COMPLETED")
    assert feature["T"]["T2"]["formula"].endswith("T_MINUS_5")
    assert feature["fixed_scores"]["internal_score"] == "SQRT_F_TIMES_T"
    assert feature["fixed_scores"]["full_mechanism_score"] == (
        "CUBE_ROOT_M_TIMES_F_TIMES_T"
    )
    mechanism = config["g2_mechanism_if_temporally_identifiable"]
    assert mechanism["macro_lead_test"]["subperiods_required_positive"] == 3
    assert mechanism["internal_bad10_test"]["minimum_scoreable_events"] == 30
    assert mechanism["internal_bad10_test"][
        "minimum_scoreable_non_event_blocks"
    ] == 120


def test_thirty_session_union_yields_exact_optimistic_origin_count() -> None:
    daily = _prepared_daily(count=40)
    assert daily["necessary_day_valid"].all()
    origins = build_potential_origin_ceiling(
        daily,
        history_sessions_inclusive=25,
        forward_sessions=5,
    )
    assert len(origins) == 11
    assert origins["potential_origin"].all()
    first = origins.iloc[0]
    assert first["necessary_start_date"] == daily.iloc[0]["date"]
    assert first["origin_date"] == daily.iloc[24]["date"]
    assert first["forward_end_date"] == daily.iloc[29]["date"]
    assert first["necessary_required_session_count"] == 30


def test_one_missing_industry_member_invalidates_overlapping_upper_bound_windows() -> None:
    membership_raw, constituent_raw, industry_raw = _synthetic_inputs(count=40)
    target_date = pd.bdate_range("2024-01-02", periods=40)[20]
    mask = industry_raw["membership_date"].eq(target_date) & industry_raw[
        "symbol"
    ].eq("000001.SZ")
    industry_raw.loc[mask, "mapping_status"] = (
        "NO_VIEW_NO_PROVABLE_RECORD_BY_MARKET_CLOSE"
    )
    membership = prepare_membership_presence(
        membership_raw,
        expected_index_code="000300",
        start=pd.Timestamp("2024-01-02"),
        cutoff=pd.Timestamp("2026-08-14"),
    )
    daily = build_daily_necessary_coverage(
        membership,
        prepare_constituent_presence(constituent_raw),
        prepare_industry_presence(
            industry_raw,
            required_mapping_status="PIT_AVAILABLE_BY_MARKET_CLOSE",
        ),
        expected_members_per_session=3,
    )
    assert not daily.loc[daily["date"].eq(target_date), "necessary_day_valid"].item()
    origins = build_potential_origin_ceiling(
        daily,
        history_sessions_inclusive=25,
        forward_sessions=5,
    )
    assert not origins["potential_origin"].any()


def _period_origin_rows(
    periods: list,
    counts: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, count in zip(periods, counts):
        dates = pd.bdate_range(period.start, periods=count)
        for date in dates:
            rows.append(
                {
                    "origin_date": date,
                    "forward_end_date": date,
                    "potential_origin": True,
                }
            )
    return pd.DataFrame(rows)


def test_gate_requires_three_of_four_periods_without_repartition() -> None:
    periods = parse_fixed_subperiods(load_config(CONFIG))
    passing = summarize_subperiod_coverage(
        _period_origin_rows(periods, [20, 20, 20, 19]),
        periods,
        minimum_potential_origins=20,
    )
    passed = adjudicate_temporal_identifiability(
        passing,
        required_evaluable_subperiods=3,
        total_subperiods=4,
    )
    assert passed["passed"] is True
    assert passed["status"] == PASS_STATUS

    failing = summarize_subperiod_coverage(
        _period_origin_rows(periods, [20, 20, 19, 19]),
        periods,
        minimum_potential_origins=20,
    )
    failed = adjudicate_temporal_identifiability(
        failing,
        required_evaluable_subperiods=3,
        total_subperiods=4,
    )
    assert failed["passed"] is False
    assert failed["status"] == FAIL_STATUS
    assert failed["terminal_for_v1"] is True
    assert failed["rescue_allowed"] is False
    assert failed["repartition_allowed"] is False


def test_preflight_result_cannot_contain_feature_label_model_or_trade_action() -> None:
    daily = _prepared_daily(count=40)
    origins = build_potential_origin_ceiling(
        daily,
        history_sessions_inclusive=25,
        forward_sessions=5,
    )
    summaries = summarize_subperiod_coverage(
        origins,
        parse_fixed_subperiods(load_config(CONFIG)),
        minimum_potential_origins=20,
    )
    gate = adjudicate_temporal_identifiability(
        summaries,
        required_evaluable_subperiods=3,
        total_subperiods=4,
    )
    result = build_preflight_result(
        subperiod_summaries=summaries,
        gate=gate,
        daily=daily,
        origins=origins,
    )
    for field in (
        "feature_values_constructed",
        "bad10_label_artifacts_read",
        "return_values_read",
        "g2_mechanism_evaluated",
        "g3_allowed",
        "model_trained",
        "probability_threshold_selected",
        "portfolio_metrics_read",
        "position_generated",
        "paper_shadow_generated",
        "order_generated",
        "broker_action_performed",
        "live_trading_authorized",
    ):
        assert result[field] is False
    assert result["position_impact"] == 0
    assert result["return_evaluation"] == "NOT_ALLOWED"
    assert result["rescue_allowed"] is False


def test_duplicate_member_identity_fails_closed() -> None:
    membership, _, _ = _synthetic_inputs(count=5)
    duplicated = pd.concat([membership, membership.iloc[[0]]], ignore_index=True)
    with pytest.raises(EvidenceContractError, match="重复"):
        prepare_membership_presence(
            duplicated,
            expected_index_code="000300",
            start=pd.Timestamp("2024-01-02"),
            cutoff=pd.Timestamp("2026-08-14"),
        )


def test_frozen_manifest_and_terminal_receipt_when_present() -> None:
    if not MANIFEST.exists():
        pytest.skip("G2 时间可识别性 manifest 尚未冻结")
    from scripts.freeze_510300_asymmetric_stress_hazard_v1_g2_identifiability import (
        verify_frozen_contract,
    )

    verify_frozen_contract(CONFIG)
    if not RECEIPT.exists():
        return
    from scripts.run_510300_asymmetric_stress_hazard_v1_g2_identifiability import (
        verify_results,
    )

    verified = verify_results(CONFIG)
    assert verified["feature_values_constructed"] is False
    assert verified["bad10_label_artifacts_read"] is False
    assert verified["portfolio_metrics_read"] is False
    assert verified["position_impact"] == 0
