from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from research import csi300_pit_fundamental_underreaction_official_facts_v1 as _base
from research import csi300_pit_fundamental_underreaction_official_facts_v1_4 as _v1_4


PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FINANCIAL_FACTS_"
    "DEFERRED_EXPENSIVE_FALLBACK_QUARTERLY_TABLE_CONTAMINATION_GUARD_V1_9_0"
)
REQUIRED_METRICS = _v1_4.REQUIRED_METRICS
ADMITTED_VERIFICATION_STATUS = _v1_4.ADMITTED_VERIFICATION_STATUS
SectionRange = _v1_4.SectionRange
ROOT = Path(__file__).resolve().parents[1]
QUARTERLY_CONTAMINATION_CORRECTION_PATH = ROOT / (
    "config/csi300_pit_fundamental_underreaction_official_facts_"
    "v1_5_quarterly_contamination_corrections.json"
)


@lru_cache(maxsize=1)
def quarterly_contamination_corrections() -> dict[str, dict[str, Any]]:
    payload = json.loads(
        QUARTERLY_CONTAMINATION_CORRECTION_PATH.read_text(encoding="utf-8")
    )
    documents = payload.get("documents") or {}
    if int(payload.get("document_count", -1)) != len(documents):
        raise RuntimeError("V1.5季度表污染修正契约文档计数不一致")
    observed_metric_count = sum(
        len(document.get("corrected_metrics") or {})
        for document in documents.values()
    )
    if int(payload.get("corrected_metric_count", -1)) != observed_metric_count:
        raise RuntimeError("V1.5季度表污染修正契约指标计数不一致")
    return {str(pdf_sha256): document for pdf_sha256, document in documents.items()}


def _quarterly_contamination_guard_receipt(
    result: dict[str, Any],
    *,
    content: bytes,
    pdf_sha256: str,
    period_type: str | None,
) -> dict[str, Any]:
    correction = quarterly_contamination_corrections().get(pdf_sha256)
    if correction is None:
        return {"status": "NOT_APPLICABLE_PDF_HASH_NOT_REGISTERED"}
    if len(content) != int(correction["official_pdf_size_bytes"]):
        raise RuntimeError("V1.5季度表污染修正契约PDF字节数不一致")
    if str(period_type or "").upper() != str(correction["period_type"]).upper():
        raise RuntimeError("V1.5季度表污染修正契约报告期类型不一致")

    metrics = {
        str(metric["metric_id"]): metric for metric in result.get("metrics") or []
    }
    verified_metric_ids: list[str] = []
    for metric_id, expected in (correction.get("corrected_metrics") or {}).items():
        metric = metrics.get(str(metric_id))
        if metric is None:
            raise RuntimeError(f"V1.5季度表污染修正契约指标缺失：{metric_id}")
        observed_value = Decimal(str(metric["metric_value_cny"]))
        expected_value = Decimal(str(expected["corrected_value_cny"]))
        if observed_value != expected_value:
            raise RuntimeError(
                f"V1.5季度表污染修正契约数值不一致：{metric_id}|"
                f"{observed_value}|{expected_value}"
            )
        allowed_pages = {int(value) for value in expected["allowed_source_pages"]}
        if int(metric["source_page"]) not in allowed_pages:
            raise RuntimeError(
                f"V1.5季度表污染修正契约证据页不一致：{metric_id}|"
                f"{metric['source_page']}|{sorted(allowed_pages)}"
            )
        verified_metric_ids.append(str(metric_id))
    return {
        "status": "PASS_HASH_BOUND_V1_4_QUARTERLY_TABLE_CONTAMINATION_CORRECTED",
        "document_key": str(correction["document_key"]),
        "official_pdf_sha256": pdf_sha256,
        "quarterly_table_page": int(correction["quarterly_table_page"]),
        "authoritative_annual_summary_page": int(
            correction["authoritative_annual_summary_page"]
        ),
        "verified_metric_ids": sorted(verified_metric_ids),
    }


def _restamp_result(result: dict[str, Any]) -> None:
    for metric in result.get("metrics") or []:
        try:
            locator = json.loads(str(metric.get("source_locator") or "{}"))
        except json.JSONDecodeError:
            locator = {"legacy_source_locator": str(metric.get("source_locator") or "")}
        previous = str(locator.get("parser_version") or result.get("parser_version") or "")
        if previous and previous != PARSER_VERSION:
            locator.setdefault("base_parser_version", previous)
        locator["parser_version"] = PARSER_VERSION
        metric["source_locator"] = json.dumps(
            locator,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metric["verification_status"] = ADMITTED_VERIFICATION_STATUS
    _v1_4._refresh_result(result)
    result["parser_version"] = PARSER_VERSION


def extract_metrics_from_page_texts(
    page_texts: Sequence[str],
    *,
    period_type: str | None = None,
    text_engine: str = "PDFIUM",
) -> dict[str, Any]:
    result = _v1_4.extract_metrics_from_page_texts(
        page_texts,
        period_type=period_type,
        text_engine=text_engine,
    )
    _restamp_result(result)
    return result


def _primary_semantic_result(
    content: bytes,
    *,
    period_type: str | None,
) -> dict[str, Any]:
    pdf_sha256 = hashlib.sha256(content).hexdigest()
    manual_entry = _v1_4._manual_appendix_documents().get(pdf_sha256)
    page_texts, text_receipt = _base.extract_pdf_page_texts(content)
    normalized = [_v1_4.normalize_financial_text(text) for text in page_texts]
    result = _v1_4.extract_metrics_from_page_texts(
        normalized,
        period_type=period_type,
        text_engine="PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
    )
    _v1_4._stamp_inherited_metrics(result)
    traditional_applied = any(
        original != converted
        for original, converted in zip(page_texts, normalized, strict=True)
    )
    sections = _v1_4.locate_statement_sections(normalized)
    pruned_before_overrides = _v1_4._prune_untrusted_statement_metrics(
        result,
        normalized,
        sections,
    )
    summary_pages = _v1_4._summary_pages(normalized, sections)
    summary_duplicates = _v1_4._add_text_summary_overrides(
        result,
        normalized,
        summary_pages,
        period_type=period_type,
    )
    _v1_4._add_logical_balance_overrides(
        result,
        normalized,
        sections["balance"],
    )
    _v1_4._add_logical_income_overrides(
        result,
        normalized,
        sections["income"],
        summary_duplicates,
        period_type=period_type,
    )
    _v1_4._add_logical_cash_override(
        result,
        normalized,
        sections["cash"],
        summary_duplicates,
        period_type=period_type,
    )
    pruned_after_overrides = _v1_4._prune_untrusted_statement_metrics(
        result,
        normalized,
        sections,
    )

    manual_receipt: dict[str, Any] = {
        "status": "NOT_APPLICABLE_PDF_HASH_NOT_REGISTERED"
    }
    if manual_entry is not None:
        manual_receipt = _v1_4._apply_manual_appendix(
            result,
            manual_entry,
            content=content,
            pdf_sha256=pdf_sha256,
        )

    result["sections"] = {
        key: {
            "name": section.name,
            "start_page": section.page_indices[0] + 1 if section.page_indices else None,
            "end_page": section.page_indices[-1] + 1 if section.page_indices else None,
            "unit": section.unit,
            "text_engine": "PDFIUM_FINANCIAL_TOKEN_NORMALIZED",
        }
        for key, section in sections.items()
    }
    result["v1_4_semantic_table_receipt"] = {
        "status": "DEFERRED_UNTIL_PRIMARY_SEMANTIC_RESULT_REMAINS_INCOMPLETE",
        "examined_page_numbers": [],
        "table_row_count": 0,
        "page_errors": [],
        "summary_duplicate_metric_ids": sorted(summary_duplicates),
        "traditional_financial_token_normalization_applied": traditional_applied,
        "pruned_untrusted_metric_ids_before_overrides": pruned_before_overrides,
        "pruned_untrusted_metric_ids_after_overrides": pruned_after_overrides,
    }
    result["hash_bound_manual_appendix_receipt"] = manual_receipt
    result["v1_3_complete_document_fallback_receipt"] = {
        "status": "DEFERRED_UNTIL_PRIMARY_SEMANTIC_RESULT_REMAINS_INCOMPLETE",
        "filled_metric_ids": [],
    }
    result["pdfplumber_secondary_text_receipt"] = {
        "status": "DEFERRED_UNTIL_PRIMARY_SEMANTIC_RESULT_REMAINS_INCOMPLETE"
    }
    result["ocr_fallback_receipt"] = {
        "status": (
            "SKIPPED_HASH_BOUND_VERIFIED_TRANSCRIPTION"
            if manual_entry is not None
            else "DEFERRED_UNTIL_PRIMARY_SEMANTIC_RESULT_REMAINS_INCOMPLETE"
        )
    }
    result.update(
        {
            "official_pdf_sha256": pdf_sha256,
            "official_pdf_size_bytes": len(content),
            **text_receipt,
        }
    )
    _restamp_result(result)
    result["v1_5_quarterly_contamination_guard_receipt"] = (
        _quarterly_contamination_guard_receipt(
            result,
            content=content,
            pdf_sha256=pdf_sha256,
            period_type=period_type,
        )
    )
    return result


def extract_official_pdf_facts(
    content: bytes,
    *,
    period_type: str | None = None,
) -> dict[str, Any]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("响应没有%PDF-文件头")

    primary = _primary_semantic_result(content, period_type=period_type)
    if primary.get("document_complete"):
        primary["v1_5_deferred_expensive_paths_receipt"] = {
            "status": "PASS_SKIPPED_EXPENSIVE_PATHS_AFTER_PRIMARY_NINE_OF_NINE",
            "primary_metric_count": len(primary.get("metrics") or []),
            "full_v1_4_path_executed": False,
        }
        return primary

    primary_missing = list(primary.get("missing_metrics") or [])
    full = _v1_4.extract_official_pdf_facts(
        content,
        period_type=period_type,
    )
    full["v1_5_deferred_expensive_paths_receipt"] = {
        "status": "PASS_EXECUTED_FULL_V1_4_PATH_AFTER_PRIMARY_INCOMPLETE",
        "primary_metric_count": len(primary.get("metrics") or []),
        "primary_missing_metrics": primary_missing,
        "full_v1_4_path_executed": True,
        "full_v1_4_document_complete": bool(full.get("document_complete")),
    }
    _restamp_result(full)
    return full


__all__ = [
    "ADMITTED_VERIFICATION_STATUS",
    "PARSER_VERSION",
    "QUARTERLY_CONTAMINATION_CORRECTION_PATH",
    "REQUIRED_METRICS",
    "SectionRange",
    "extract_metrics_from_page_texts",
    "extract_official_pdf_facts",
    "quarterly_contamination_corrections",
]
