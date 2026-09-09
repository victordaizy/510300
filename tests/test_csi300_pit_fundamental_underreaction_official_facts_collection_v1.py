from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from scripts.collect_csi300_pit_fundamental_underreaction_official_facts_v1 import (
    REQUIRED_METRICS,
    build_document_queue,
    build_event_dependency_ledger,
    build_requirement_ledger,
    standard_quarter_dependencies,
)


def make_config(expected_target_count: int) -> dict:
    return {
        "inputs": {
            "target_inventory": {
                "target_flag": "included_in_phase1_fact_target",
                "expected_target_event_count": expected_target_count,
            }
        },
        "dependency_contract": {
            "quarters_including_target": 9,
            "missing_dependency_status": "NO_VIEW_REQUIRED_OFFICIAL_PERIODIC_REPORT_MISSING",
        },
        "official_pdf": {"accepted_host": "static.cninfo.com.cn"},
        "artifacts": {
            "checkpoint_root": "tmp/test_official_fact_collection/checkpoints"
        },
    }


def report_period(year: int, quarter: int) -> date:
    month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = month_day[quarter]
    return date(year, month, day)


def period_type(quarter: int) -> str:
    return {1: "Q1", 2: "H1", 3: "Q3", 4: "FY"}[quarter]


def event_row(
    year: int,
    quarter: int,
    *,
    publication_date: date | None = None,
) -> dict:
    period = report_period(year, quarter)
    announcement_id = f"A{year}{quarter}"
    published = publication_date or period + timedelta(days=30)
    return {
        "announcement_id": announcement_id,
        "ts_code": "000001.SZ",
        "sec_code": "000001",
        "sec_name": "测试公司",
        "announcement_title": f"{year}年第{quarter}季度报告",
        "period_type": period_type(quarter),
        "report_period": period,
        "event_publication_date": published,
        "official_timestamp_at": pd.Timestamp(published, tz="Asia/Shanghai"),
        "official_pdf_url": (
            f"https://static.cninfo.com.cn/finalpage/{published.isoformat()}/{announcement_id}.PDF"
        ),
        "relative_pdf_url": f"finalpage/{published.isoformat()}/{announcement_id}.PDF",
        "org_id": "org-test",
        "adjunct_size_kb": 100,
    }


def target_row(year: int, quarter: int, publication_date: date) -> dict:
    event = event_row(year, quarter, publication_date=publication_date)
    return {
        "announcement_id": event["announcement_id"],
        "event_publication_date": publication_date,
        "official_pdf_url": event["official_pdf_url"],
        "period_type": event["period_type"],
        "report_period": event["report_period"],
        "ts_code": event["ts_code"],
        "industry_l1": "测试行业",
        "industry_l1_code": "801000.SI",
        "publication_year": publication_date.year,
        "included_in_phase1_fact_target": True,
    }


def events_from_2022_q3_to_2024_q4() -> pd.DataFrame:
    rows = []
    for absolute_quarter in range(2022 * 4 + 2, 2024 * 4 + 4):
        year = absolute_quarter // 4
        quarter = absolute_quarter % 4 + 1
        rows.append(event_row(year, quarter))
    return pd.DataFrame(rows)


def test_standard_quarter_dependencies_are_current_plus_eight_q_dec_quarters() -> None:
    dependencies = standard_quarter_dependencies("2024-09-30", "Q3")
    assert len(dependencies) == 9
    assert dependencies[0] == {
        "dependency_lag_quarters": 0,
        "dependency_role": "TARGET_REPORT",
        "required_report_period": "2024-09-30",
        "required_period_type": "Q3",
    }
    assert dependencies[-1]["required_report_period"] == "2022-09-30"
    assert dependencies[-1]["required_period_type"] == "Q3"


def test_requirement_ledger_never_uses_dependency_published_after_target() -> None:
    target_date = date(2024, 10, 31)
    inventory = pd.DataFrame([target_row(2024, 3, target_date)])
    current = event_row(2024, 3, publication_date=target_date)
    late_prior = event_row(2024, 2, publication_date=date(2024, 11, 1))
    ledger = build_requirement_ledger(
        inventory,
        pd.DataFrame([current, late_prior]),
        make_config(1),
    )
    assert len(ledger) == 9
    assert ledger.loc[
        ledger["required_report_period"].eq("2024-09-30"),
        "dependency_temporal_status",
    ].item() == "PASS_OFFICIAL_DEPENDENCY_AVAILABLE_AT_TARGET"
    assert ledger.loc[
        ledger["required_report_period"].eq("2024-06-30"),
        "dependency_temporal_status",
    ].item() == "NO_VIEW_REQUIRED_OFFICIAL_REPORT_NOT_PUBLIC_AT_TARGET"
    assert int(ledger["queued_for_official_pdf"].sum()) == 1


def test_overlapping_target_dependencies_are_downloaded_once() -> None:
    q3_publication = date(2024, 10, 31)
    fy_publication = date(2025, 3, 31)
    inventory = pd.DataFrame(
        [
            target_row(2024, 3, q3_publication),
            target_row(2024, 4, fy_publication),
        ]
    )
    events = events_from_2022_q3_to_2024_q4()
    events.loc[events["announcement_id"].eq("A20243"), "event_publication_date"] = (
        q3_publication
    )
    events.loc[events["announcement_id"].eq("A20244"), "event_publication_date"] = (
        fy_publication
    )
    ledger = build_requirement_ledger(inventory, events, make_config(2))
    queue = build_document_queue(ledger, make_config(2))
    assert len(ledger) == 18
    assert len(queue) == 10
    assert queue["announcement_id"].is_unique
    assert queue["dependent_target_event_count"].max() == 2


def test_target_ready_requires_all_nine_documents_and_all_metrics() -> None:
    target_date = date(2024, 10, 31)
    inventory = pd.DataFrame([target_row(2024, 3, target_date)])
    events = events_from_2022_q3_to_2024_q4()
    events.loc[events["announcement_id"].eq("A20243"), "event_publication_date"] = (
        target_date
    )
    ledger = build_requirement_ledger(inventory, events, make_config(1))
    queue = build_document_queue(ledger, make_config(1))
    queue_state = queue.assign(
        checkpoint_status="PARSED_COMPLETE",
        document_terminal=True,
        document_complete=True,
        missing_metric_count=0,
        missing_metrics_json="[]",
    )
    _, dependency = build_event_dependency_ledger(ledger, queue_state)
    assert len(dependency) == 1
    assert bool(dependency.iloc[0]["target_event_ready"]) is True
    assert int(dependency.iloc[0]["required_document_count"]) == 9
    assert int(dependency.iloc[0]["complete_document_count"]) == 9

    one_id = queue_state.iloc[0]["announcement_id"]
    queue_state.loc[
        queue_state["announcement_id"].eq(one_id),
        ["checkpoint_status", "document_complete", "missing_metric_count"],
    ] = ["PARSED_INCOMPLETE", False, len(REQUIRED_METRICS)]
    _, incomplete_dependency = build_event_dependency_ledger(ledger, queue_state)
    assert bool(incomplete_dependency.iloc[0]["target_event_ready"]) is False
    assert int(incomplete_dependency.iloc[0]["incomplete_document_count"]) == 1
    assert int(incomplete_dependency.iloc[0]["missing_metric_count"]) == len(
        REQUIRED_METRICS
    )
