from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.a_share_hs_partial_cash_tender_20x20_source_v1 import (
    PartialTenderSourceError,
    build_lifecycle_candidate_ledger,
    build_machine_source_ledger,
    canonical_frame_sha256_payload,
    classify_lifecycle_roles,
    extract_offer_period_candidates,
    extract_offer_price_candidates,
    extract_offer_ratio_candidates,
    validate_source_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
TEXT_REVIEW = (
    ROOT
    / "data/staging/a_share_hs_official_unconditional_full_cash_tender_spread_v1"
    / "formal_report_text_review_v1.parquet"
)
ADJUDICATION = TEXT_REVIEW.with_name("formal_report_adjudication_ledger_v1.parquet")
METADATA = ROOT / "data/raw/cninfo/a_share_hs_cash_tender_offer_metadata_v1.parquet"


@pytest.fixture(scope="module")
def official_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(TEXT_REVIEW),
        pd.read_parquet(ADJUDICATION),
        pd.read_parquet(METADATA),
    )


@pytest.fixture(scope="module")
def machine_source(
    official_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    text_review, adjudication, _ = official_inputs
    return build_machine_source_ledger(text_review, adjudication)


def test_extracts_appendix_offer_quantity_ratio() -> None:
    text = "预定收购股份数量：100,000,000股，约占总股本，比例：20.00%"
    candidates = extract_offer_ratio_candidates(text)

    assert len(candidates) == 1
    assert candidates[0]["ratio_percent"] == pytest.approx(20.0)
    assert "APPENDIX_QUANTITY_AND_RATIO" in candidates[0]["pattern_ids"]


def test_extracts_narrative_offer_quantity_ratio() -> None:
    text = "本次要约收购股份数量为123,000股，占公司总股本的27.49%。"
    candidates = extract_offer_ratio_candidates(text)

    assert len(candidates) == 1
    assert candidates[0]["ratio_percent"] == pytest.approx(27.49)
    assert "NARRATIVE_OFFER_QUANTITY_TOTAL_SHARE_RATIO" in candidates[0][
        "pattern_ids"
    ]


def test_price_candidate_parser_deduplicates_and_rejects_non_array() -> None:
    assert extract_offer_price_candidates('[10.2, "11.3", 10.2, -1, null]') == [
        10.2,
        11.3,
    ]
    with pytest.raises(PartialTenderSourceError, match="JSON数组"):
        extract_offer_price_candidates('{"price": 10.2}')


def test_period_parser_accepts_valid_chinese_dates_only() -> None:
    value = json.dumps(
        [
            ["2017年2月22日", "2017年3月23日"],
            ["2017年2月30日", "2017年3月23日"],
            ["2017年3月23日", "2017年2月22日"],
        ],
        ensure_ascii=False,
    )

    assert extract_offer_period_candidates(value) == [
        ["2017-02-22", "2017-03-23"]
    ]


def test_lifecycle_title_roles_are_deterministic() -> None:
    assert classify_lifecycle_roles("关于要约收购结果暨股份过户完成的公告") == [
        "COMPLETION",
        "RESULT",
    ]
    assert classify_lifecycle_roles("关于终止要约收购的公告") == ["TERMINATION"]
    assert classify_lifecycle_roles("与要约事项有关的提示性公告") == [
        "OTHER_OFFER_RELATED"
    ]


def test_real_official_inputs_validate_exact_source_scope(
    official_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    text_review, adjudication, metadata = official_inputs
    result = validate_source_inputs(text_review, adjudication, metadata)

    assert result["row_counts"] == {
        "formal_report_text_review": 111,
        "formal_report_adjudication_ledger": 111,
        "cninfo_metadata_archive": 680,
    }
    assert result["partial_formal_report_rows"] == 76
    assert result["partial_unique_security_count"] == 65
    assert result["partial_calendar_year_count"] == 10
    assert result["partial_subject_market_price_read"] is False
    assert result["partial_subject_future_return_read"] is False
    assert result["partial_tender_outcome_read"] is False


def test_real_source_build_has_76_rows_and_no_market_or_outcome_read(
    machine_source: pd.DataFrame,
) -> None:
    assert len(machine_source) == 76
    assert machine_source["announcement_id"].is_unique
    assert machine_source["ts_code"].nunique() == 65
    assert not machine_source["market_price_read"].any()
    assert not machine_source["future_return_read"].any()
    assert not machine_source["tender_outcome_read"].any()
    forbidden = {
        "raw_open",
        "raw_close",
        "gross_offer_premium",
        "future_return",
        "acceptance_fraction",
        "accepted_share_count",
        "event_return",
        "benchmark_return",
        "excess_return",
    }
    assert not (forbidden & set(machine_source.columns))


def test_known_20_percent_case_and_minimum_condition_case_are_preserved(
    machine_source: pd.DataFrame,
) -> None:
    indexed = machine_source.set_index("announcement_id")
    normal = indexed.loc["1203095079"]
    assert normal["ts_code"] == "601028.SH"
    assert normal["machine_offer_quantity_ratio"] == pytest.approx(0.20)
    assert not normal["minimum_acceptance_condition_detected"]
    assert normal["machine_offer_price_cny"] == pytest.approx(10.39)
    assert normal["machine_offer_start_date"] == "2017-02-22"
    assert normal["machine_offer_end_date"] == "2017-03-23"
    assert normal["machine_status"] == (
        "POTENTIAL_20X20_SOURCE_EVENT_MANUAL_CONFIRMATION_REQUIRED"
    )

    conditioned = indexed.loc["1204106498"]
    assert conditioned["ts_code"] == "000403.SZ"
    assert conditioned["machine_offer_quantity_ratio"] == pytest.approx(0.2749)
    assert conditioned["minimum_acceptance_condition_detected"]
    reasons = json.loads(conditioned["machine_exclusion_reasons_json"])
    assert "MINIMUM_ACCEPTANCE_CONDITION_DETECTED" in reasons


def test_lifecycle_candidates_are_source_only_and_start_at_formal_report(
    official_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    machine_source: pd.DataFrame,
) -> None:
    _, _, metadata = official_inputs
    lifecycle = build_lifecycle_candidate_ledger(metadata, machine_source)

    assert len(lifecycle) == 290
    assert lifecycle["ts_code"].nunique() == 65
    assert (
        lifecycle["official_pdf_date"] >= lifecycle["first_partial_formal_pdf_date"]
    ).all()
    assert not lifecycle["market_price_read"].any()
    assert not lifecycle["future_return_read"].any()
    assert not lifecycle["tender_outcome_read"].any()
    roles = lifecycle["candidate_roles_json"].map(json.loads).explode()
    assert int(roles.eq("RESULT").sum()) == 59
    assert int(roles.eq("COMPLETION").sum()) == 23
    assert int(roles.eq("TERMINATION").sum()) == 2


def test_source_validation_hard_fails_if_partial_market_flag_is_true(
    official_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    text_review, adjudication, metadata = official_inputs
    changed = adjudication.copy()
    partial_index = changed.index[
        changed["reason_code"].astype(str).str.contains("PARTIAL", regex=False)
    ][0]
    changed.loc[partial_index, "market_price_read"] = True

    with pytest.raises(PartialTenderSourceError, match="提前读取禁止字段"):
        validate_source_inputs(text_review, changed, metadata)


def test_canonical_payload_is_stable_for_an_identical_rebuild(
    official_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    text_review, adjudication, _ = official_inputs
    first = build_machine_source_ledger(text_review, adjudication)
    second = build_machine_source_ledger(text_review, adjudication)

    assert canonical_frame_sha256_payload(first) == canonical_frame_sha256_payload(
        second
    )
