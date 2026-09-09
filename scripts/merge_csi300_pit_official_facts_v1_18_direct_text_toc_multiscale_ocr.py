from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import merge_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as merge  # noqa: E402


PROTOCOL_ID = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_18_DIRECT_TEXT_TOC_MULTISCALE_OCR_MERGE_V1"
)
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_18_DIRECT_TEXT_TOC_MULTISCALE_OCR_"
    "MERGED_V1_0_0"
)
FIRST_PASS_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_15_multiscale_note_bridge_direct_pdf_ocr/receipt.json"
)
ZERO_SELECTION_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_18_zero_selection_direct_text_toc_ocr/receipt.json"
)
COMBINED_ROOT = ROOT / "tmp/pit_v1_18_combined_direct_text_toc_multiscale_ocr"
COMBINED_RECEIPT_PATH = COMBINED_ROOT / "receipt.json"
ADAPTED_RECEIPT_PATH = COMBINED_ROOT / "merge_input_receipt.json"


def _sha256_file(path: Path) -> str:
    return merge.prior._sha256_file(path)


def _load_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("target_exception_count") != 0
        or receipt.get("source_exception_count") != 0
        or receipt.get("market_price_read") is not False
        or receipt.get("future_return_read") is not False
        or receipt.get("return_evaluation") != "NOT_ALLOWED"
    ):
        raise RuntimeError(f"OCR收据异常或触碰收益读取边界：{path}")
    return receipt


def _row_map(receipt: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    rows = list(receipt.get("results") or [])
    result = {str(row["announcement_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{label}收据公告键重复")
    return result


def _selected_page_count(row: dict[str, Any]) -> int:
    return sum(
        int(
            (source.get("selection_receipt") or {}).get("selected_page_count")
            or 0
        )
        for source in row.get("source_results") or []
    )


def _combine_receipts() -> dict[str, Any]:
    first = _load_receipt(FIRST_PASS_RECEIPT_PATH)
    zero = _load_receipt(ZERO_SELECTION_RECEIPT_PATH)
    first_map = _row_map(first, label="首轮")
    zero_map = _row_map(zero, label="零选页补跑")
    if not set(zero_map).issubset(first_map):
        raise RuntimeError("零选页补跑目标超出首轮集合")

    for announcement_id, replacement in zero_map.items():
        original = first_map[announcement_id]
        if _selected_page_count(original) != 0:
            raise RuntimeError(f"替换目标并非首轮零选页：{announcement_id}")
        if original.get("newly_admitted_metric_ids") or original.get("new_metrics"):
            raise RuntimeError(f"首轮零选页目标已有新增指标：{announcement_id}")
        if list(original.get("after_missing_metrics") or []) != list(
            replacement.get("before_missing_metrics") or []
        ):
            raise RuntimeError(f"补跑前后缺失状态不衔接：{announcement_id}")
        if (
            str(original.get("checkpoint_path"))
            != str(replacement.get("checkpoint_path"))
            or str(original.get("checkpoint_sha256"))
            != str(replacement.get("checkpoint_sha256"))
        ):
            raise RuntimeError(f"补跑检查点基线漂移：{announcement_id}")

    results = [
        zero_map.get(announcement_id, first_map[announcement_id])
        for announcement_id in sorted(first_map)
    ]
    combined = {
        "protocol_id": (
            "CSI300_PIT_OFFICIAL_FACTS_V1_18_DIRECT_TEXT_TOC_"
            "MULTISCALE_OCR_COMBINED_RECEIPT_V1"
        ),
        "source_receipts": [
            {
                "path": str(FIRST_PASS_RECEIPT_PATH.relative_to(ROOT)),
                "sha256": _sha256_file(FIRST_PASS_RECEIPT_PATH),
            },
            {
                "path": str(ZERO_SELECTION_RECEIPT_PATH.relative_to(ROOT)),
                "sha256": _sha256_file(ZERO_SELECTION_RECEIPT_PATH),
            },
        ],
        "first_pass_target_count": len(first_map),
        "zero_selection_replacement_count": len(zero_map),
        "target_count": len(results),
        "newly_completed_document_count": sum(
            bool(row.get("document_complete_after_targeted_ocr"))
            and not bool(row.get("document_complete_before_targeted_ocr"))
            for row in results
        ),
        "newly_admitted_metric_count": sum(
            len(row.get("newly_admitted_metric_ids") or []) for row in results
        ),
        "selected_page_count": sum(_selected_page_count(row) for row in results),
        "target_exception_count": 0,
        "source_exception_count": 0,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    merge.prior._atomic_write_json(COMBINED_RECEIPT_PATH, combined)
    return combined


def _adapt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "merged_checkpoint_path": str(row["checkpoint_path"]),
        "merged_checkpoint_sha256": str(row["checkpoint_sha256"]),
        "changed_metric_ids": list(row.get("newly_admitted_metric_ids") or []),
        "changed_metrics": list(row.get("new_metrics") or []),
        "corrected_metric_ids": [],
        "removed_metric_ids": [],
    }


def _build_merge_input_receipt() -> None:
    combined = _combine_receipts()
    adapted = {
        **combined,
        "source_combined_receipt_path": str(COMBINED_RECEIPT_PATH.relative_to(ROOT)),
        "source_combined_receipt_sha256": _sha256_file(COMBINED_RECEIPT_PATH),
        "removed_unreplaced_metric_count": 0,
        "results": [_adapt_row(row) for row in combined["results"]],
    }
    merge.prior._atomic_write_json(ADAPTED_RECEIPT_PATH, adapted)


def main() -> int:
    _build_merge_input_receipt()
    audit_root = merge.AUDIT_ROOT
    raw_root = merge.RAW_ROOT
    merge.PROTOCOL_ID = PROTOCOL_ID
    merge.MERGED_PARSER_VERSION = MERGED_PARSER_VERSION
    merge.ADMISSION_PHASE = "DIRECT_TEXT_TOC_MULTISCALE_OCR"
    merge.MERGE_RECEIPT_FIELD = "direct_text_toc_multiscale_ocr_merge_receipt"
    merge.BASE_RECEIPT_PATH = raw_root / "receipt_v1_9_receivable_direct_pdf.json"
    merge.RESIDUAL_RECEIPT_PATH = ADAPTED_RECEIPT_PATH
    merge.CHECKPOINT_ROOT = (
        raw_root / "checkpoints_v1_18_direct_text_toc_multiscale_ocr_merged"
    )
    merge.OUTPUT_REQUIREMENT_PATH = (
        audit_root
        / "official_fact_requirement_ledger_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_QUEUE_PATH = (
        audit_root
        / "official_fact_document_queue_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_PATCH_MANIFEST_PATH = (
        audit_root
        / "official_fact_checkpoint_patch_manifest_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_RESIDUAL_PRIORITY_PATH = (
        audit_root
        / "official_fact_residual_priority_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_FACTS_PATH = (
        raw_root
        / "official_financial_facts_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_DEPENDENCY_PATH = (
        raw_root
        / "event_fact_dependency_ledger_v1_18_direct_text_toc_multiscale_ocr.parquet"
    )
    merge.OUTPUT_RECEIPT_PATH = (
        raw_root / "receipt_v1_18_direct_text_toc_multiscale_ocr.json"
    )
    merge.OUTPUT_REPORT_PATH = (
        ROOT
        / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
        "OFFICIAL_FACTS_V1_18_DIRECT_TEXT_TOC_MULTISCALE_OCR.md"
    )
    return merge.main()


if __name__ == "__main__":
    raise SystemExit(main())
