"""沪深300点时成分 2015 条件扩展的回归测试。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.csi300_pit_membership_weights_source_remediation_v1 import (
    OfficialCycle,
    assert_outcome_blind_columns,
    load_official_2015_extension_cycles,
    load_unique_open_dates,
    resolve_membership_transitions,
)


ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "curated" / "510300_csi300_pit_membership_weights_source_remediation_v1"
SOURCE_MANIFEST = CURATED / "official_2015_extension_manifest.json"
CLOCK_ADDENDUM = (
    ROOT
    / "config"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_0_2_2015_extension_clock_addendum.json"
)
CALENDAR = (
    ROOT
    / "data"
    / "staging"
    / "a_share_hs_concentrated_low_risk_trend_v1_1"
    / "trading_calendar_observed_open_days.parquet"
)
EXTENSION_DAILY = CURATED / "000300_daily_pit_membership_20150101_20160612.parquet"
MAIN_DAILY = CURATED / "000300_daily_pit_membership_20160613_20260814.parquet"
COMBINED_DAILY = CURATED / "000300_daily_pit_membership_20150101_20260814.parquet"
SNAPSHOT_COMPARISON = CURATED / "official_2015_extension_snapshot_comparison.parquet"
ADMISSION = CURATED / "official_2015_extension_admission_manifest.json"
INDEPENDENT_AUDIT = (
    ROOT
    / "reports"
    / "audit"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_independent_audit.json"
)
AUTHORITATIVE_STATUS = ROOT / "reports" / "research" / "510300_authoritative_research_status_v1.json"


@pytest.fixture(scope="module")
def official_cycles() -> list[OfficialCycle]:
    return load_official_2015_extension_cycles(ROOT, SOURCE_MANIFEST)


def test_2015_official_source_chain_is_complete_and_reparsed(
    official_cycles: list[OfficialCycle],
) -> None:
    assert [cycle.cycle_id for cycle in official_cycles] == [
        "2015_SPECIAL_HYSEC_MERGER",
        "2015_SPECIAL_CNR_OPG_DELISTING",
        "2015_H1_REGULAR",
        "2015_H2_REGULAR",
        "2015_SPECIAL_CHINA_MERCHANTS_PROPERTY_MERGER",
    ]
    assert [cycle.effective_date for cycle in official_cycles] == [
        date(2015, 1, 26),
        date(2015, 5, 20),
        date(2015, 6, 15),
        date(2015, 12, 14),
        date(2015, 12, 30),
    ]
    assert [len(cycle.additions) for cycle in official_cycles] == [1, 2, 18, 20, 1]
    assert [len(cycle.deletions) for cycle in official_cycles] == [1, 2, 18, 20, 1]
    assert official_cycles[0].additions == frozenset({"000166.SZ"})
    assert official_cycles[0].deletions == frozenset({"000562.SZ"})
    assert official_cycles[1].additions == frozenset({"300003.SZ", "000738.SZ"})
    assert official_cycles[1].deletions == frozenset({"601299.SH", "600832.SH"})
    assert official_cycles[-1].additions == frozenset({"001979.SZ"})
    assert official_cycles[-1].deletions == frozenset({"000024.SZ"})

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    classifications = manifest["announcement_search_classification"]
    assert len(classifications) == 14
    assert sum(item["classification"].startswith("INCLUDED") for item in classifications) == 5
    assert sum(item["classification"].startswith("EXCLUDED") for item in classifications) == 8
    assert sum(item["classification"].startswith("HANDOVER") for item in classifications) == 1
    sse_dates = {
        item["security_code"]: item["delisting_date"]
        for item in manifest["source_objects"]
        if item["kind"] == "SSE_OFFICIAL_DELISTING_QUERY"
    }
    assert sse_dates == {"601299": "2015-05-20", "600832": "2015-05-20"}
    assert manifest["market_price_read"] is False
    assert manifest["future_return_read"] is False
    assert manifest["portfolio_evaluation"] == "NOT_ALLOWED"


def test_2015_clock_addendum_maps_every_cycle_to_an_open_session(
    official_cycles: list[OfficialCycle],
) -> None:
    open_dates = load_unique_open_dates(CALENDAR)
    transitions = resolve_membership_transitions(
        ROOT,
        official_cycles,
        open_dates,
        CLOCK_ADDENDUM,
    )
    assert len(transitions) == 5
    assert [item.transition_session for item in transitions] == [
        date(2015, 1, 26),
        date(2015, 5, 20),
        date(2015, 6, 15),
        date(2015, 12, 14),
        date(2015, 12, 30),
    ]
    assert all(item.transition_session in set(open_dates) for item in transitions)


def test_2015_extension_and_combined_daily_panels_are_exact() -> None:
    extension = pd.read_parquet(EXTENSION_DAILY)
    main = pd.read_parquet(MAIN_DAILY)
    combined = pd.read_parquet(COMBINED_DAILY)
    for frame in (extension, main, combined):
        assert_outcome_blind_columns(frame.columns)
        frame["membership_date"] = pd.to_datetime(frame["membership_date"]).dt.date
        assert not frame.duplicated(["membership_date", "symbol"]).any()
        assert frame.groupby("membership_date")["symbol"].nunique().eq(300).all()

    assert min(extension["membership_date"]) == date(2015, 1, 5)
    assert max(extension["membership_date"]) == date(2016, 6, 8)
    assert extension["membership_date"].nunique() == 350
    assert len(extension) == 105_000
    assert min(main["membership_date"]) == date(2016, 6, 13)
    assert max(combined["membership_date"]) == date(2026, 8, 14)
    assert combined["membership_date"].nunique() == 2_823
    assert len(combined) == 846_900

    reconstructed = pd.concat([extension, main], ignore_index=True).sort_values(
        ["membership_date", "symbol"]
    ).reset_index(drop=True)
    observed = combined.sort_values(["membership_date", "symbol"]).reset_index(drop=True)
    assert reconstructed.equals(observed)


def test_official_snapshot_handover_and_status_boundaries() -> None:
    comparisons = pd.read_parquet(SNAPSHOT_COMPARISON)
    comparisons["snapshot_date"] = pd.to_datetime(comparisons["snapshot_date"]).dt.date
    assert set(comparisons["snapshot_date"]) == {
        date(2015, 3, 2),
        date(2015, 7, 1),
        date(2015, 9, 1),
    }
    assert comparisons["set_difference_count"].eq(0).all()
    assert set(comparisons["role"]) == {"REPLAY_ANCHOR", "INDEPENDENT_OFFICIAL_CHECK"}

    admission = json.loads(ADMISSION.read_text(encoding="utf-8"))
    assert admission["membership_admission"]["status"] == (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION"
    )
    assert admission["membership_admission"]["handover_set_difference_count"] == 0
    assert admission["membership_admission"]["deterministic_replay_match"] is True
    assert admission["weight_admission"]["status"] == (
        "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    )
    assert admission["weight_admission"]["version_or_as_of_provenance_gate_pass"] is False
    assert admission["candidate_boundary"]["return_evaluation"] == "NOT_ALLOWED"
    assert admission["candidate_boundary"]["model_action"] == "ABSTAIN"
    assert admission["trading_authorization"] is False

    audit = json.loads(INDEPENDENT_AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_MEMBERSHIP_ADMISSION_WEIGHT_BLOCK_PRESERVED"
    assert audit["membership"]["handover_set_difference_count"] == 0
    assert audit["weight"]["version_or_as_of_provenance_gate_pass"] is False
    assert audit["market_price_read"] is False
    assert audit["future_return_read"] is False
    assert audit["portfolio_evaluation"] == "NOT_ALLOWED"

    authoritative = json.loads(AUTHORITATIVE_STATUS.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in authoritative["research_objects"]
        if item["name"] == "510300_PIT_EARNINGS_INFORMATION_DIFFUSION_10D_RISK_V1"
    )
    assert candidate["status"] == "BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS"
    assert candidate["membership_remediation_status"] == (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_2015_EXTENSION"
    )
    assert candidate["remaining_data_blocker"] == (
        "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    )
    assert candidate["future_outcome_read_allowed"] is False
    assert candidate["portfolio_evaluation_allowed"] is False
    assert candidate["tradable"] is False
