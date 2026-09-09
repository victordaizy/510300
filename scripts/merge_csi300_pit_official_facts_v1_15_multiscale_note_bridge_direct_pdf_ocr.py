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
    "CSI300_PIT_OFFICIAL_FACTS_V1_15_MULTISCALE_NOTE_BRIDGE_"
    "DIRECT_PDF_OCR_MERGE_V1"
)
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_15_MULTISCALE_NOTE_BRIDGE_"
    "DIRECT_PDF_OCR_MERGED_V1_0_0"
)
OCR_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_15_multiscale_note_bridge_direct_pdf_ocr/receipt.json"
)
ADAPTED_RECEIPT_PATH = (
    ROOT
    / "tmp/pit_v1_15_multiscale_note_bridge_direct_pdf_ocr/"
    "merge_input_receipt.json"
)


def _sha256_file(path: Path) -> str:
    return merge.prior._sha256_file(path)


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
    receipt = json.loads(OCR_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        receipt.get("target_exception_count") != 0
        or receipt.get("source_exception_count") != 0
        or receipt.get("market_price_read") is not False
        or receipt.get("future_return_read") is not False
    ):
        raise RuntimeError("多倍率OCR收据存在异常或触碰收益读取边界")
    adapted = {
        **receipt,
        "source_ocr_receipt_path": str(OCR_RECEIPT_PATH.relative_to(ROOT)),
        "source_ocr_receipt_sha256": _sha256_file(OCR_RECEIPT_PATH),
        "removed_unreplaced_metric_count": 0,
        "results": [_adapt_row(row) for row in receipt.get("results") or []],
    }
    merge.prior._atomic_write_json(ADAPTED_RECEIPT_PATH, adapted)


def main() -> int:
    _build_merge_input_receipt()
    audit_root = merge.AUDIT_ROOT
    raw_root = merge.RAW_ROOT
    merge.PROTOCOL_ID = PROTOCOL_ID
    merge.MERGED_PARSER_VERSION = MERGED_PARSER_VERSION
    merge.ADMISSION_PHASE = "MULTISCALE_NOTE_BRIDGE_DIRECT_PDF_OCR"
    merge.MERGE_RECEIPT_FIELD = (
        "multiscale_note_bridge_direct_pdf_ocr_merge_receipt"
    )
    merge.BASE_RECEIPT_PATH = raw_root / "receipt_v1_9_receivable_direct_pdf.json"
    merge.RESIDUAL_RECEIPT_PATH = ADAPTED_RECEIPT_PATH
    merge.CHECKPOINT_ROOT = (
        raw_root / "checkpoints_v1_15_multiscale_note_bridge_ocr_merged"
    )
    merge.OUTPUT_REQUIREMENT_PATH = (
        audit_root
        / "official_fact_requirement_ledger_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_QUEUE_PATH = (
        audit_root
        / "official_fact_document_queue_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_PATCH_MANIFEST_PATH = (
        audit_root
        / "official_fact_checkpoint_patch_manifest_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_RESIDUAL_PRIORITY_PATH = (
        audit_root
        / "official_fact_residual_priority_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_FACTS_PATH = (
        raw_root
        / "official_financial_facts_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_DEPENDENCY_PATH = (
        raw_root
        / "event_fact_dependency_ledger_v1_15_multiscale_note_bridge_ocr.parquet"
    )
    merge.OUTPUT_RECEIPT_PATH = (
        raw_root / "receipt_v1_15_multiscale_note_bridge_direct_pdf_ocr.json"
    )
    merge.OUTPUT_REPORT_PATH = (
        ROOT
        / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
        "OFFICIAL_FACTS_V1_15_MULTISCALE_NOTE_BRIDGE_DIRECT_PDF_OCR.md"
    )
    return merge.main()


if __name__ == "__main__":
    raise SystemExit(main())
