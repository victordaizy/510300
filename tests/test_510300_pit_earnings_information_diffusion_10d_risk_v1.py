from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from research.pit_earnings_information_diffusion_10d_risk_v1 import (
    active_member_counts_on_dates,
    build_member_event_audit,
    choose_status,
    validate_membership_intervals,
    validate_weights,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_pit_earnings_information_diffusion_10d_risk_v1.yaml"
MANIFEST = ROOT / "config/510300_pit_earnings_information_diffusion_10d_risk_v1_manifest.json"


def test_config_is_outcome_blind_and_abstains() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["protocol"]["version"] == "1.0.2"
    assert config["protocol"]["research_stage"] == "DATA_FEASIBILITY_ONLY"
    assert config["protocol"]["future_outcome_reads_allowed"] is False
    assert config["protocol"]["prediction_model_allowed"] is False
    assert config["protocol"]["portfolio_evaluation_allowed"] is False
    assert config["point_in_time_contract"]["force_common_2015_start_allowed"] is False
    assert config["governance"]["model_action"] == "ABSTAIN"
    assert config["governance"]["model_position_target"] == "UNSET"
    assert config["governance"]["live_trading_authorized"] is False


def test_membership_and_strict_prior_weight_rule() -> None:
    metadata = pd.DataFrame(
        {
            "announcement_id": ["1", "2"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "announcement_type": ["EARNINGS_FORECAST", "EARNINGS_EXPRESS"],
            "target_title_type": [True, True],
            "announcement_timestamp_at": pd.to_datetime(
                ["2020-01-31 00:00:00+08:00", "2020-02-03 16:00:00+08:00"]
            ),
            "status": ["PASS_OFFICIAL_METADATA_CAPTURED"] * 2,
            "market_price_read": [False, False],
            "future_return_read": [False, False],
        }
    )
    inventory = pd.DataFrame(
        {
            "announcement_id": ["1", "2"],
            "pdf_status": ["PASS_REUSED_VERIFIED_PDF"] * 2,
            "pdf_sha256": ["a" * 64, "b" * 64],
            "market_price_read": [False, False],
            "future_return_read": [False, False],
        }
    )
    intervals = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "opt_in": pd.to_datetime(["2019-01-01"]),
            "opt_out": [pd.NaT],
            "source": ["TEST_INTERVAL"],
            "source_license": ["TEST"],
        }
    )
    weights = pd.DataFrame(
        {
            "index_code": ["399300.SZ", "399300.SZ"],
            "con_code": ["000001.SZ", "000001.SZ"],
            "trade_date": pd.to_datetime(["2019-12-31", "2020-01-31"]),
            "weight": [1.0, 1.1],
            "source": ["TEST_WEIGHT", "TEST_WEIGHT"],
            "retrieved_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    trading_calendar = pd.DataFrame(
        {
            "exchange": ["SSE", "SZSE", "SSE", "SZSE"],
            "date": pd.to_datetime(
                ["2020-02-03", "2020-02-03", "2020-02-04", "2020-02-04"]
            ),
            "is_open": [True, True, True, True],
            "available_at": pd.to_datetime(
                ["2020-02-03"] * 2 + ["2020-02-04"] * 2
            ),
            "source": ["TEST_CALENDAR"] * 4,
        }
    )
    events, summary = build_member_event_audit(
        metadata,
        inventory,
        intervals,
        weights,
        trading_calendar,
        maximum_snapshot_age_days=45,
        maximum_availability_lag_days=7,
    )
    first = events.loc[events["announcement_id"].eq("1")].iloc[0]
    second = events.loc[events["announcement_id"].eq("2")].iloc[0]
    assert first["weight_snapshot_date"] == pd.Timestamp("2019-12-31")
    assert first["weight"] == 1.0
    assert second["weight_snapshot_date"] == pd.Timestamp("2020-01-31")
    assert second["weight"] == 1.1
    assert first["availability_date"] == pd.Timestamp("2020-02-03")
    assert second["availability_date"] == pd.Timestamp("2020-02-04")
    assert events["future_return_read"].sum() == 0
    assert events["market_price_read"].sum() == 0
    assert summary["first_valid_strict_prior_weight_event_date"].startswith("2020-01-31")


def test_structural_validators_and_active_count() -> None:
    intervals = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "opt_in": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "opt_out": [pd.NaT, pd.NaT],
            "source": ["S", "S"],
            "source_license": ["L", "L"],
        }
    )
    summary = validate_membership_intervals(intervals)
    assert summary["duplicate_symbol_opt_in_rows"] == 0
    assert summary["overlapping_interval_rows"] == 0
    counts = active_member_counts_on_dates(intervals, [pd.Timestamp("2020-01-02")])
    assert counts.iloc[0] == 2

    weights = pd.DataFrame(
        {
            "index_code": ["I", "I"],
            "con_code": ["A", "B"],
            "trade_date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "weight": [50.0, 50.0],
            "source": ["S", "S"],
            "retrieved_at": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        }
    )
    _, weight_summary = validate_weights(weights)
    assert weight_summary["duplicate_date_security_rows"] == 0
    assert weight_summary["weight_sum_min_percent"] == 100.0


def test_status_machine_preserves_running_and_pending() -> None:
    statuses = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["status_machine"]
    running = choose_status(
        activation_passed=True,
        metadata_passed=True,
        membership_weights_passed=True,
        facts_progress_status="RUNNING_MULTIPROCESS_FACT_CHECKPOINT_EXECUTION",
        facts_automation_status=None,
        facts_automation_passed=None,
        human_review_completed=False,
        facts_prevalence_passed=None,
        statuses=statuses,
    )
    assert running == "RUNNING_UPSTREAM_FIRST_PUBLIC_FACT_EXTRACTION"
    pending = choose_status(
        activation_passed=True,
        metadata_passed=True,
        membership_weights_passed=True,
        facts_progress_status="PASS_MULTIPROCESS_FACT_TASKS_ATTEMPTED_WITH_EXPLICIT_RESULTS",
        facts_automation_status=(
            "PASS_OFFICIAL_TARGET_PDF_AND_CORE_FACT_AUTOMATION_V1_HUMAN_REVIEW_PENDING"
        ),
        facts_automation_passed=True,
        human_review_completed=False,
        facts_prevalence_passed=True,
        statuses=statuses,
    )
    assert pending == "PENDING_DUAL_INDEPENDENT_HUMAN_REVIEW"


def test_manifest_hashes_match_if_manifest_exists() -> None:
    if not MANIFEST.is_file():
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "FROZEN_OUTCOME_BLIND_PIT_EARNINGS_DATA_FEASIBILITY_"
        "BEFORE_ANY_POST_EVENT_OUTCOME"
    )
    for section in ("files", "fixed_input_evidence"):
        for item in manifest[section]:
            path = ROOT / item["path"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
