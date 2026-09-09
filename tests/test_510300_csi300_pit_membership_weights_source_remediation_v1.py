"""沪深300点时成分与权重来源修复 V1 的回归测试。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research.csi300_pit_membership_weights_source_remediation_v1 import (
    MembershipTransition,
    OfficialCycle,
    assert_outcome_blind_columns,
    build_daily_membership_panel,
    compare_replay_to_snapshots,
    derive_past_anchor_from_transitions,
    load_current_official_anchor,
    load_official_cycles,
    load_official_special_cycles,
    load_unique_open_dates,
    load_weight_snapshots,
    normalize_symbol,
    replay_membership_transitions,
    resolve_membership_transitions,
    state_on_or_before,
)


ROOT = Path(__file__).resolve().parents[1]
REGULAR_MANIFEST = (
    ROOT
    / "data"
    / "curated"
    / "a_share_hs_csi300_official_addition_forced_demand_v1"
    / "official_announcement_manifest.json"
)
SPECIAL_MANIFEST = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "official_special_rebalance_manifest.json"
)
CLOCK_CORRECTION = (
    ROOT
    / "config"
    / "510300_csi300_pit_membership_weights_source_remediation_v1_0_1_clock_correction.json"
)
CALENDAR = (
    ROOT
    / "data"
    / "staging"
    / "a_share_hs_concentrated_low_risk_trend_v1_1"
    / "trading_calendar_observed_open_days.parquet"
)
CURRENT_ANCHOR = ROOT / "data" / "raw" / "constituents" / "000300_current_weights.parquet"
HISTORICAL_WEIGHTS = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
DAILY_MEMBERSHIP = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "000300_daily_pit_membership_20160613_20260814.parquet"
)
ADMISSION_MANIFEST = (
    ROOT
    / "data"
    / "curated"
    / "510300_csi300_pit_membership_weights_source_remediation_v1"
    / "source_remediation_admission_manifest.json"
)


def test_symbol_normalization_and_outcome_blind_column_guard() -> None:
    assert normalize_symbol("600000") == "600000.SH"
    assert normalize_symbol("000001") == "000001.SZ"
    assert normalize_symbol("688981.SH") == "688981.SH"
    assert_outcome_blind_columns(
        ["exchange", "date", "is_open", "available_at", "source", "membership_date"]
    )
    with pytest.raises(ValueError, match="禁止字段"):
        assert_outcome_blind_columns(["symbol", "future_return_10d"])
    with pytest.raises(ValueError, match="禁止字段"):
        assert_outcome_blind_columns(["symbol", "market_close_price"])


def test_synthetic_transition_round_trip_and_daily_panel() -> None:
    anchor = frozenset({"000001.SZ", "600000.SH"})
    cycle = OfficialCycle(
        cycle_id="SYNTHETIC",
        announcement_id=1,
        announcement_date=date(2021, 1, 4),
        effective_date=date(2021, 1, 5),
        additions=frozenset({"600519.SH"}),
        deletions=frozenset({"600000.SH"}),
        source_format="SYNTHETIC",
        extraction_method="SYNTHETIC",
        detail_path="synthetic.json",
        detail_sha256="0" * 64,
        attachment_paths=(),
        attachment_sha256s=(),
    )
    transition = MembershipTransition(
        cycle=cycle,
        transition_session=date(2021, 1, 6),
        clock_rule="FIRST_UNIQUE_OPEN_DATE_STRICTLY_AFTER_OFFICIAL_STATED_CLOSE_DATE",
    )
    states, audits = replay_membership_transitions(
        date(2021, 1, 4),
        anchor,
        [transition],
    )
    assert audits[0]["missing_deletions"] == []
    assert audits[0]["existing_additions"] == []
    assert state_on_or_before(states, date(2021, 1, 5)) == anchor
    assert state_on_or_before(states, date(2021, 1, 6)) == frozenset(
        {"000001.SZ", "600519.SH"}
    )

    final_set = state_on_or_before(states, date(2021, 1, 6))
    recovered, backward = derive_past_anchor_from_transitions(
        date(2021, 1, 6),
        final_set,
        date(2021, 1, 4),
        [transition],
    )
    assert recovered == anchor
    assert backward[0]["missing_additions"] == []
    assert backward[0]["existing_deletions"] == []

    large_anchor = frozenset(f"{index:06d}.SZ" for index in range(300))
    panel = build_daily_membership_panel(
        [date(2021, 1, 4), date(2021, 1, 5)],
        {date(2021, 1, 4): large_anchor},
        date(2021, 1, 4),
        date(2021, 1, 5),
        {},
    )
    assert len(panel) == 600
    assert panel.groupby("membership_date")["symbol"].nunique().eq(300).all()


def test_actual_official_transition_clock_and_monthly_reconciliation() -> None:
    regular = load_official_cycles(ROOT, REGULAR_MANIFEST)
    special = load_official_special_cycles(ROOT, SPECIAL_MANIFEST)
    open_dates = load_unique_open_dates(CALENDAR)
    transitions = resolve_membership_transitions(
        ROOT,
        sorted(regular + special, key=lambda item: item.effective_date),
        open_dates,
        CLOCK_CORRECTION,
    )
    assert len(regular) == 21
    assert len(special) == 3
    assert len(transitions) == 24
    session_by_cycle = {
        item.cycle.cycle_id: item.transition_session for item in transitions
    }
    assert session_by_cycle["2021H1"] == date(2021, 6, 15)
    assert session_by_cycle["2021H2"] == date(2021, 12, 13)
    assert session_by_cycle["2025_SPECIAL_HAITONG_DELISTING"] == date(2025, 3, 4)
    assert session_by_cycle["2025_SPECIAL_CSIC_DELISTING"] == date(2025, 9, 5)

    current_date, current_set, _ = load_current_official_anchor(CURRENT_ANCHOR)
    extension_set, backward = derive_past_anchor_from_transitions(
        current_date,
        current_set,
        date(2016, 6, 13),
        transitions,
    )
    states, forward = replay_membership_transitions(
        date(2016, 6, 13),
        extension_set,
        transitions,
    )
    assert len(extension_set) == 300
    assert all(not item["missing_additions"] for item in backward)
    assert all(not item["existing_deletions"] for item in backward)
    assert all(not item["missing_deletions"] for item in forward)
    assert all(not item["existing_additions"] for item in forward)
    assert state_on_or_before(states, current_date) == current_set

    _, snapshots = load_weight_snapshots(HISTORICAL_WEIGHTS)
    comparisons = compare_replay_to_snapshots(
        states,
        snapshots,
        date(2016, 8, 31),
        date(2026, 7, 31),
    )
    assert len(comparisons) == 120
    assert all(item["set_difference_count"] == 0 for item in comparisons)


def test_admitted_daily_membership_and_status_boundaries() -> None:
    frame = pd.read_parquet(DAILY_MEMBERSHIP)
    assert_outcome_blind_columns(frame.columns)
    frame["membership_date"] = pd.to_datetime(frame["membership_date"]).dt.date
    primary = frame.loc[frame["membership_date"].map(lambda value: value >= date(2021, 1, 1))]
    counts = primary.groupby("membership_date")["symbol"].nunique()
    assert len(counts) == 1361
    assert len(primary) == 408_300
    assert counts.eq(300).all()
    assert not frame.duplicated(["membership_date", "symbol"]).any()

    manifest = json.loads(ADMISSION_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["membership_admission"]["status"] == (
        "PASS_OFFICIAL_PIT_CSI300_MEMBERSHIP_PRIMARY_2021_WINDOW_ONLY_WEIGHTS_PENDING"
    )
    assert manifest["weight_admission"]["structural_gate_pass"] is True
    assert manifest["weight_admission"]["version_or_as_of_provenance_gate_pass"] is False
    assert manifest["weight_admission"]["status"] == (
        "BLOCKED_NO_VERSION_PROVEN_PIT_CSI300_WEIGHTS"
    )
    assert manifest["candidate_boundary"]["model_action"] == "ABSTAIN"
    assert manifest["candidate_boundary"]["portfolio_evaluation"] == "NOT_ALLOWED"
    assert manifest["market_price_read"] is False
    assert manifest["future_return_read"] is False
    assert manifest["trading_authorization"] is False
