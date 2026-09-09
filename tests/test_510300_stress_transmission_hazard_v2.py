from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.stress_transmission_hazard_v2 import (
    ACTION_NONE_CONFIRMED,
    ACTION_RESOLVED,
    ACTION_UNRESOLVED,
    NO_VIEW,
    PROGRAM_ID,
    UNASSIGNED_SPLIT,
    VIEW_ALLOWED,
    ConstituentReturnState,
    StressTransmissionContractError,
    assert_predictive_names_are_industry_free,
    assign_group_splits,
    build_bad10_event_ledger,
    build_daily_coverage_ledger,
    build_internal_raw_features,
    causal_latest_observation,
    causal_midrank,
    classify_constituent_returns,
    fit_constrained_logit,
    validate_event_weighting,
    validate_split_integrity,
)
from scripts.freeze_510300_stress_transmission_hazard_v2 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _return_input(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": "2020-01-02",
        "symbol": "000001.SZ",
        "previous_unadjusted_close": 10.0,
        "unadjusted_close": 11.0,
        "source_observed": True,
        "supplier_conflict": False,
        "official_suspension": False,
        "suspension_evidence_id": "",
        "corporate_action_status": ACTION_NONE_CONFIRMED,
        "corporate_action_evidence_id": "ACTION_CHECK_20200102",
        "cash_distribution_per_pre_event_share": 0.0,
        "post_to_pre_share_ratio": 1.0,
        "subscription_cash_outflow_per_pre_event_share": 0.0,
    }
    row.update(overrides)
    return row


def test_four_state_classifier_never_conflates_missing_with_zero() -> None:
    rows = pd.DataFrame(
        [
            _return_input(symbol="000001.SZ"),
            _return_input(
                symbol="000002.SZ",
                unadjusted_close=np.nan,
                source_observed=False,
                official_suspension=True,
                suspension_evidence_id="SSE_SUSPENSION_20200102_000002",
            ),
            _return_input(
                symbol="000003.SZ",
                corporate_action_status=ACTION_UNRESOLVED,
                corporate_action_evidence_id="ACTION_PENDING_000003",
            ),
            _return_input(
                symbol="000004.SZ",
                previous_unadjusted_close=np.nan,
                source_observed=True,
            ),
        ]
    )
    result = classify_constituent_returns(rows).set_index("symbol")

    assert result.loc["000001.SZ", "constituent_return_state"] == (
        ConstituentReturnState.TRADED_VALID.value
    )
    assert result.loc["000001.SZ", "daily_total_shareholder_return"] == pytest.approx(
        0.10
    )
    assert result.loc["000002.SZ", "constituent_return_state"] == (
        ConstituentReturnState.OFFICIAL_SUSPENSION.value
    )
    assert result.loc["000002.SZ", "daily_total_shareholder_return"] == 0.0
    assert result.loc["000003.SZ", "constituent_return_state"] == (
        ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value
    )
    assert pd.isna(result.loc["000003.SZ", "daily_total_shareholder_return"])
    assert result.loc["000004.SZ", "constituent_return_state"] == (
        ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value
    )
    assert pd.isna(result.loc["000004.SZ", "daily_total_shareholder_return"])


def test_official_suspension_requires_evidence_and_reconciled_action() -> None:
    missing_evidence = classify_constituent_returns(
        pd.DataFrame(
            [
                _return_input(
                    official_suspension=True,
                    source_observed=False,
                    unadjusted_close=np.nan,
                    suspension_evidence_id="",
                )
            ]
        )
    ).iloc[0]
    assert missing_evidence["constituent_return_state"] == (
        ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value
    )
    assert pd.isna(missing_evidence["daily_total_shareholder_return"])

    unresolved = classify_constituent_returns(
        pd.DataFrame(
            [
                _return_input(
                    official_suspension=True,
                    source_observed=False,
                    unadjusted_close=np.nan,
                    suspension_evidence_id="SSE_SUSPENSION",
                    corporate_action_status=ACTION_UNRESOLVED,
                )
            ]
        )
    ).iloc[0]
    assert unresolved["constituent_return_state"] == (
        ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value
    )
    assert pd.isna(unresolved["daily_total_shareholder_return"])


def test_resolved_corporate_action_uses_complete_wealth_identity() -> None:
    result = classify_constituent_returns(
        pd.DataFrame(
            [
                _return_input(
                    corporate_action_status=ACTION_RESOLVED,
                    corporate_action_evidence_id="CNINFO_ACTION_001",
                    unadjusted_close=5.0,
                    post_to_pre_share_ratio=2.0,
                    cash_distribution_per_pre_event_share=0.2,
                    subscription_cash_outflow_per_pre_event_share=0.1,
                )
            ]
        )
    ).iloc[0]
    assert result["constituent_return_state"] == ConstituentReturnState.TRADED_VALID.value
    assert result["daily_total_shareholder_return"] == pytest.approx(0.01)


def test_none_confirmed_with_non_neutral_action_terms_fails_closed() -> None:
    result = classify_constituent_returns(
        pd.DataFrame(
            [
                _return_input(
                    corporate_action_status=ACTION_NONE_CONFIRMED,
                    cash_distribution_per_pre_event_share=0.1,
                )
            ]
        )
    ).iloc[0]
    assert result["constituent_return_state"] == (
        ConstituentReturnState.CORPORATE_ACTION_UNRESOLVED.value
    )
    assert pd.isna(result["daily_total_shareholder_return"])


def test_supplier_conflict_has_fail_closed_precedence() -> None:
    result = classify_constituent_returns(
        pd.DataFrame(
            [
                _return_input(
                    supplier_conflict=True,
                    official_suspension=True,
                    suspension_evidence_id="CONFLICTING_SUSPENSION_SOURCE",
                )
            ]
        )
    ).iloc[0]
    assert result["constituent_return_state"] == (
        ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value
    )
    assert pd.isna(result["daily_total_shareholder_return"])


def _classified_for_coverage(
    *,
    count: int,
    usable_count: int,
    date: str = "2020-01-02",
) -> pd.DataFrame:
    symbols = [f"{number:06d}.SZ" for number in range(1, count + 1)]
    usable = np.arange(count) < usable_count
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "symbol": symbols,
            "constituent_return_state": np.where(
                usable,
                ConstituentReturnState.TRADED_VALID.value,
                ConstituentReturnState.SUPPLIER_MISSING_OR_CONFLICT.value,
            ),
            "daily_total_shareholder_return": np.where(usable, 0.001, np.nan),
            "return_is_usable": usable,
        }
    )


def _membership(count: int, date: str = "2020-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "symbol": [f"{number:06d}.SZ" for number in range(1, count + 1)],
            "index_code": "000300",
        }
    )


def test_member_coverage_gate_passes_at_98_percent_and_fails_below() -> None:
    members = _membership(100)
    passed = build_daily_coverage_ledger(
        membership=members,
        classified_returns=_classified_for_coverage(count=100, usable_count=98),
        expected_members_per_day=100,
    ).iloc[0]
    failed = build_daily_coverage_ledger(
        membership=members,
        classified_returns=_classified_for_coverage(count=100, usable_count=97),
        expected_members_per_day=100,
    ).iloc[0]
    assert passed["usable_member_ratio"] == pytest.approx(0.98)
    assert passed["aggregation_state"] == VIEW_ALLOWED
    assert failed["aggregation_state"] == NO_VIEW


def test_reliable_weight_coverage_gate_is_exactly_99_percent() -> None:
    members = _membership(100)
    members["weight"] = 0.01
    passed = build_daily_coverage_ledger(
        membership=members,
        classified_returns=_classified_for_coverage(count=100, usable_count=99),
        reliable_point_in_time_weights=True,
        expected_members_per_day=100,
    ).iloc[0]
    failed_returns = _classified_for_coverage(count=100, usable_count=98)
    failed = build_daily_coverage_ledger(
        membership=members,
        classified_returns=failed_returns,
        reliable_point_in_time_weights=True,
        expected_members_per_day=100,
    ).iloc[0]
    assert passed["usable_weight_ratio"] == pytest.approx(0.99)
    assert passed["aggregation_state"] == VIEW_ALLOWED
    assert failed["aggregation_state"] == NO_VIEW


def test_causal_midrank_is_invariant_to_future_append() -> None:
    initial = pd.Series([1.0, 3.0, 2.0, 4.0])
    extended = pd.concat([initial, pd.Series([-100.0, 100.0])], ignore_index=True)
    initial_rank = causal_midrank(initial)
    extended_rank = causal_midrank(extended).iloc[: len(initial)]
    pd.testing.assert_series_equal(initial_rank.reset_index(drop=True), extended_rank)
    assert np.isnan(initial_rank.iloc[0])
    assert initial_rank.iloc[2] == pytest.approx(0.5)


def test_causal_latest_observation_does_not_use_after_close_release() -> None:
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    releases = pd.DataFrame(
        {
            "observation_date": ["2020-01-02", "2020-01-03"],
            "available_at": [
                "2020-01-02T06:00:00Z",
                "2020-01-03T08:00:00Z",
            ],
            "value": [1.0, 2.0],
        }
    )
    result = causal_latest_observation(
        market_dates=dates,
        releases=releases,
        value_column="value",
    )
    assert result["value"].tolist() == [1.0, 1.0]
    assert result["value_source_observation_date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-02"),
    ]


def test_internal_features_are_industry_free_and_use_leave_one_out_peers() -> None:
    dates = pd.bdate_range("2020-01-02", periods=80)
    symbols = [f"{number:06d}.SZ" for number in range(1, 11)]
    membership_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    market_returns: list[float] = []
    for date_index, date in enumerate(dates):
        daily_values: list[float] = []
        common = 0.002 * np.sin(date_index / 3.0)
        for symbol_index, symbol in enumerate(symbols):
            value = common + 0.0003 * np.cos((date_index + symbol_index) / 5.0)
            daily_values.append(value)
            membership_rows.append(
                {"date": date, "symbol": symbol, "index_code": "000300"}
            )
            return_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "constituent_return_state": ConstituentReturnState.TRADED_VALID.value,
                    "daily_total_shareholder_return": value,
                    "return_is_usable": True,
                }
            )
        market_returns.append(float(np.mean(daily_values)))
    benchmark_close = 1000.0 * np.cumprod(1.0 + np.asarray(market_returns))
    features = build_internal_raw_features(
        membership=pd.DataFrame(membership_rows),
        classified_returns=pd.DataFrame(return_rows),
        h00300_total_return_close=pd.DataFrame(
            {"date": dates, "close": benchmark_close}
        ),
        expected_members_per_day=10,
    )
    assert not any(
        token in column.casefold()
        for column in features.columns
        for token in ("industry", "sector", "shenwan")
    )
    last = features.iloc[-1]
    assert last["comovement_scoreable_member_ratio"] >= 0.90
    assert last["internal_coverage_state"] == VIEW_ALLOWED
    assert np.isfinite(last["F"])


def _origin(
    origin_date: str,
    entry_date: str,
    horizon_end_date: str,
    bad10: int,
) -> dict[str, object]:
    return {
        "origin_date": pd.Timestamp(origin_date),
        "entry_date": pd.Timestamp(entry_date),
        "horizon_end_date": pd.Timestamp(horizon_end_date),
        "bad10": bad10,
        "first_breach_date": pd.Timestamp(entry_date) if bad10 else pd.NaT,
    }


def test_bad10_event_ledger_uses_transitive_overlap_and_unit_event_weight() -> None:
    origin_panel = pd.DataFrame(
        [
            _origin("2020-01-02", "2020-01-03", "2020-01-16", 1),
            _origin("2020-01-08", "2020-01-09", "2020-01-22", 1),
            _origin("2020-01-20", "2020-01-21", "2020-02-03", 1),
            _origin("2020-02-10", "2020-02-11", "2020-02-24", 0),
            _origin("2020-03-02", "2020-03-03", "2020-03-16", 1),
        ]
    )
    events, samples = build_bad10_event_ledger(origin_panel)
    assert events["event_id"].tolist() == [
        "BAD10_V2_EVENT_0001",
        "BAD10_V2_EVENT_0002",
    ]
    assert events["positive_origin_count"].tolist() == [3, 1]
    first_weights = samples.loc[
        samples["event_id"].eq("BAD10_V2_EVENT_0001"), "training_weight"
    ]
    assert first_weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    validate_event_weighting(events=events, samples=samples)
    assert samples["split_assignment"].eq(UNASSIGNED_SPLIT).all()


def test_split_assignment_operates_on_whole_event_and_validator_rejects_leakage() -> None:
    origin_panel = pd.DataFrame(
        [
            _origin("2020-01-02", "2020-01-03", "2020-01-16", 1),
            _origin("2020-01-08", "2020-01-09", "2020-01-22", 1),
            _origin("2020-02-10", "2020-02-11", "2020-02-24", 0),
        ]
    )
    _, samples = build_bad10_event_ledger(origin_panel)
    event_group = "BAD10_V2_EVENT_0001"
    assigned = assign_group_splits(
        samples,
        {
            event_group: "TRAIN",
            "NON_EVENT_DAY_20200210": "EVALUATION",
        },
    )
    assert assigned.loc[assigned["bad10"].eq(1), "split_assignment"].eq("TRAIN").all()

    leaked = assigned.copy()
    positive_indices = leaked.index[leaked["bad10"].eq(1)].tolist()
    leaked.loc[positive_indices[0], "split_assignment"] = "CALIBRATION"
    with pytest.raises(StressTransmissionContractError, match="跨越多个样本区间"):
        validate_split_integrity(leaked)


def test_constrained_logit_has_nonnegative_coefficients_and_fixed_l2() -> None:
    matrix = np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.1, 0.2],
            [0.7, 0.5, 0.4],
            [0.9, 0.8, 0.7],
            [0.8, 0.9, 0.6],
            [0.3, 0.2, 0.1],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 1, 0])
    weights = np.ones(len(labels))
    fit = fit_constrained_logit(
        model_id="B3",
        matrix=matrix,
        labels=labels,
        sample_weight=weights,
    )
    assert fit.converged is True
    assert fit.l2_penalty == 1.0
    assert all(coefficient >= 0.0 for coefficient in fit.coefficients)
    with pytest.raises(StressTransmissionContractError, match="禁止搜索"):
        fit_constrained_logit(
            model_id="B3",
            matrix=matrix,
            labels=labels,
            sample_weight=weights,
            l2_penalty=0.5,
        )


def test_industry_semantics_are_rejected_from_predictive_names() -> None:
    assert_predictive_names_are_industry_free(["T", "T_x_F", "T_x_M"])
    with pytest.raises(StressTransmissionContractError, match="行业分类语义"):
        assert_predictive_names_are_industry_free(["sw_industry_corr20"])


def test_protocol_freezes_research_only_scope_and_no_performance_read() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    assert config["program"]["program_id"] == PROGRAM_ID
    assert config["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["program"]["position_impact"] == 0
    assert config["industry_contract"]["predictive_use"] == "NOT_ALLOWED"
    assert config["constituent_return_contract"][
        "blanket_fill_missing_return_with_zero_allowed"
    ] is False
    assert config["model_contract"]["l2_penalty"] == 1.0
    assert config["model_contract"]["auc_role"] == "DESCRIPTION_ONLY"
    assert config["mechanical_position_experiment_reserved_for_post_G4"][
        "probability_threshold"
    ] is None


def test_v2_core_contains_no_blanket_fillna_zero_expression() -> None:
    source = (ROOT / "research/stress_transmission_hazard_v2.py").read_text(
        encoding="utf-8"
    )
    compact = source.replace(" ", "")
    assert ".fillna(0" not in compact


def test_frozen_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        pytest.skip("V2 manifest 尚未生成")
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["program_id"] == PROGRAM_ID
    assert manifest["return_evaluation"] == "NOT_ALLOWED"
    assert manifest["portfolio_results_read"] is False
