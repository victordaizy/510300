"""全发布序列逐行来源资格，与原完整快照裁决分别记录。"""
from __future__ import annotations

import pandas as pd

from research import official_industry_classification_inputs_v1 as previous


def qualify_metadata(row):
    result = dict(row)
    structural = bool(row.get("title_passed") and row.get("rows", 0) > 0
        and row.get("duplicate_security_rows") == 0 and not row.get("unparsed_text_codes")
        and not row.get("codes_absent_from_text") and row.get("page_clock_verified", True))
    result["complete_snapshot_passed"] = bool(row.get("parse_passed"))
    result["row_source_snapshot_eligible"] = structural
    result["source_role"] = "明确证券行的来源用途；原整表裁决单列，未知行不补。"
    return result


def latest_published(metadata, decision):
    decision = pd.Timestamp(decision)
    if decision.tzinfo is None:
        decision = decision.tz_localize("Asia/Shanghai")
    known = [row for row in metadata if previous.publication_clock(row["published"]) <= decision]
    return max(known, key=lambda row: (previous.publication_clock(row["published"]), row["id"])) if known else None


def qualified_rows(frame, metadata):
    result = frame.copy()
    if not len(result):
        result["row_source_known"] = pd.Series(dtype="bool")
        return result
    source_codes = result.section_code.fillna("").str.fullmatch(r"[A-Z]") & result.major_code.fillna("").str.fullmatch(r"\d{2}")
    lineage = result[["row_origin", "section_origin", "major_origin"]].notna().all(axis=1)
    names = result[["section_name", "major_name"]].notna().all(axis=1)
    result["row_source_known"] = result.classification_known.eq(True) & source_codes & lineage & names & metadata["row_source_snapshot_eligible"]
    return result
