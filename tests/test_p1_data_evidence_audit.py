from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_p1_data_evidence import (
    _extract_csi300_codes,
    _extract_csi300_changes,
    build_membership_events,
    calendar_day_difference,
    canonical_rows_sha256,
    deterministic_financial_sample,
    expected_regular_effective_date,
)


def test_extract_csi300_codes_tolerates_mojibake_titles(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return (
                "乱码标题\n"
                "000001 平安银行 600000 浦发银行\n"
                "000002 万科A 600001 示例公司\n"
                "另一乱码标题\n"
                "000003 示例公司 600002 示例公司\n"
            )

    class FakeDocument:
        pages = [FakePage()]

        def __enter__(self) -> "FakeDocument":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "scripts.audit_p1_data_evidence.pdfplumber.open",
        lambda _: FakeDocument(),
    )
    assert _extract_csi300_codes(b"%PDF-fake") == {
        "000001",
        "000002",
        "600000",
        "600001",
    }
    assert _extract_csi300_changes(b"%PDF-fake") == (
        {"000001", "000002"},
        {"600000", "600001"},
    )


def test_expected_regular_effective_date_uses_next_trading_day() -> None:
    trading_dates = pd.to_datetime(["2024-12-12", "2024-12-13", "2024-12-16"])
    assert expected_regular_effective_date(2024, 12, trading_dates) == pd.Timestamp(
        "2024-12-16"
    )
    with pytest.raises(ValueError, match="只适用于6月或12月"):
        expected_regular_effective_date(2024, 9, trading_dates)


def test_calendar_day_difference_accepts_aware_and_naive_dates() -> None:
    retrieved = pd.Series(pd.to_datetime(["2026-08-19T02:51:02+08:00"]))
    published = pd.Series(pd.to_datetime(["2026-08-11"]))
    assert calendar_day_difference(retrieved, published).iloc[0] == 8


def test_membership_event_transition_is_balanced_and_hashed() -> None:
    membership = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "opt_in": "2019-01-01", "opt_out": "2020-06-15"},
            {"symbol": "000002.SZ", "opt_in": "2019-01-01", "opt_out": None},
            {"symbol": "000003.SZ", "opt_in": "2020-06-15", "opt_out": None},
        ]
    )
    events = build_membership_events(membership)
    assert len(events) == 1
    assert events.loc[0, "opt_in_count"] == 1
    assert events.loc[0, "opt_out_count"] == 1
    assert len(events.loc[0, "transition_sha256"]) == 64


def test_canonical_snapshot_hash_is_row_order_invariant() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-07-31", "code": "000002.SZ", "weight": 2.0},
            {"date": "2026-07-31", "code": "000001.SZ", "weight": 1.0},
        ]
    )
    columns = ["date", "code", "weight"]
    assert canonical_rows_sha256(frame, columns) == canonical_rows_sha256(
        frame.iloc[::-1], columns
    )


def _financial_fixture() -> pd.DataFrame:
    rows = []
    for year in (2020, 2021, 2022):
        for industry_index, industry in enumerate(("银行", "医药生物", "电子")):
            for period_type, suffix in (("Q1", "0331"), ("H1", "0630"), ("ANNUAL", "1231")):
                for index in range(10):
                    rows.append(
                        {
                            "con_code": f"{year % 100:02d}{industry_index}{index:03d}.SZ",
                            "report_period": pd.Timestamp(f"{year}-{suffix[:2]}-{suffix[2:]}"),
                            "available_at": pd.Timestamp(f"{year + 1}-04-30")
                            + pd.Timedelta(days=index),
                            "report_year": year,
                            "period_type": period_type,
                            "industry_l1": industry,
                            "income_report_type_label": "1" if index else "MISSING",
                            "balance_report_type_label": "1" if index else "MISSING",
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates(
        ["con_code", "report_period", "available_at"]
    )


def test_financial_sample_is_deterministic_and_covers_strata() -> None:
    frame = _financial_fixture()
    first = deterministic_financial_sample(frame, target_size=200)
    second = deterministic_financial_sample(
        frame.sample(frac=1.0, random_state=7), target_size=200
    )
    assert first["audit_record_id"].tolist() == second["audit_record_id"].tolist()
    assert set(first["report_year"]) == set(frame["report_year"])
    assert set(first["industry_l1"]) == set(frame["industry_l1"])
    assert set(first["period_type"]) == set(frame["period_type"])


def test_financial_sample_rejects_less_than_200() -> None:
    with pytest.raises(ValueError, match="不得低于200"):
        deterministic_financial_sample(_financial_fixture(), target_size=199)


def test_generated_membership_evidence_remains_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "reports"
            / "data_quality"
            / "csi300_point_in_time_membership_evidence_20260819.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "BLOCKED_POINT_IN_TIME_MEMBERSHIP"
    assert report["membership"]["fully_official_event_count"] == 0
    assert report["weights"]["official_historical_snapshot_verified_count"] == 0
    assert report["governance"]["raw_inputs_modified"] is False


def test_generated_financial_sample_exposes_vintage_limits() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = pd.read_csv(
        root
        / "reports"
        / "audit"
        / "csi300_financial_vintage_sample_20260819.csv"
    )
    required = {
        "retrieved_at",
        "original_publication_at",
        "revision_possible",
        "vintage_verified",
    }
    assert len(sample) >= 200
    assert required <= set(sample.columns)
    assert sample["retrieved_at"].notna().all()
    assert sample["original_publication_at"].notna().all()
    assert sample["revision_possible"].all()
    assert not sample["vintage_verified"].any()
