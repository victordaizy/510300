from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from research.a_share_hs_cninfo_periodic_report_metadata_v1 import normalize_records


EVENT_COLUMNS = [
    "announcement_id",
    "ts_code",
    "sec_code",
    "sec_name",
    "org_id",
    "page_column",
    "announcement_title",
    "period_type",
    "report_period",
    "event_publication_date",
    "official_timestamp_at",
    "official_timestamp_date",
    "official_internal_date_equal",
    "relative_pdf_url",
    "official_pdf_url",
    "adjunct_type",
    "adjunct_size_kb",
    "query_interval",
    "retrieved_at",
]


def normalize_records_with_chronology(
    records: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    retrieved_at: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_records, _ = normalize_records(
        records,
        config=config,
        retrieved_at=retrieved_at,
    )
    if all_records.empty:
        empty_events = pd.DataFrame(columns=EVENT_COLUMNS)
        return all_records, empty_events, all_records.copy()

    all_records = all_records.copy()
    base_eligible = (
        all_records["title_status"].eq("ACCEPTED_ORIGINAL_FULL")
        & all_records["ts_code"].notna()
        & all_records["official_internal_date_equal"]
        & all_records["adjunct_type"].str.upper().eq("PDF")
    )
    chronology_invalid = (
        base_eligible
        & all_records["official_url_date"].notna()
        & all_records["report_period"].notna()
        & all_records["official_url_date"].lt(all_records["report_period"])
    )
    all_records["chronology_status"] = "NOT_APPLICABLE_TO_ACCEPTED_EVENT"
    all_records.loc[
        base_eligible & ~chronology_invalid, "chronology_status"
    ] = "ACCEPTED_EVENT_CHRONOLOGY_VALID"
    all_records.loc[
        chronology_invalid, "chronology_status"
    ] = "REJECTED_EVENT_PUBLICATION_BEFORE_REPORT_PERIOD"

    invalid = all_records.loc[chronology_invalid].copy()
    accepted = all_records.loc[base_eligible & ~chronology_invalid].copy()
    accepted = accepted.sort_values(
        [
            "ts_code",
            "report_period",
            "official_url_date",
            "announcement_time_ms",
            "announcement_id",
        ],
        kind="stable",
    )
    events = accepted.drop_duplicates(["ts_code", "report_period"], keep="first").copy()
    events = events.rename(columns={"official_url_date": "event_publication_date"})
    return (
        all_records,
        events[EVENT_COLUMNS].reset_index(drop=True),
        invalid.reset_index(drop=True),
    )
