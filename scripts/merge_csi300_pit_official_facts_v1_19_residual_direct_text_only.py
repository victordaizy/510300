from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import merge_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as merge  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_V1_19_RESIDUAL_DIRECT_TEXT_ONLY_MERGE_V1"
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_19_RESIDUAL_DIRECT_TEXT_ONLY_MERGED_V1_0_0"
)
SOURCE_RECEIPT_PATH = ROOT / "tmp/pit_v1_19_all_residual_direct_text_only/receipt.json"
ADAPTED_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_19_all_residual_direct_text_only/merge_input_receipt.json"
)


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
    receipt = json.loads(SOURCE_RECEIPT_PATH.read_text(encoding="utf-8"))
    if (
        receipt.get("target_exception_count") != 0
        or receipt.get("source_exception_count") != 0
        or receipt.get("market_price_read") is not False
        or receipt.get("future_return_read") is not False
        or receipt.get("return_evaluation") != "NOT_ALLOWED"
        or receipt.get("direct_text_only") is not True
    ):
        raise RuntimeError("残缺文本直读收据异常或触碰收益读取边界")
    adapted = {
        **receipt,
        "source_direct_text_receipt_path": str(SOURCE_RECEIPT_PATH.relative_to(ROOT)),
        "source_direct_text_receipt_sha256": merge.prior._sha256_file(
            SOURCE_RECEIPT_PATH
        ),
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
    merge.ADMISSION_PHASE = "RESIDUAL_DIRECT_TEXT_ONLY"
    merge.MERGE_RECEIPT_FIELD = "residual_direct_text_only_merge_receipt"
    merge.BASE_RECEIPT_PATH = (
        raw_root / "receipt_v1_18_direct_text_toc_multiscale_ocr.json"
    )
    merge.RESIDUAL_RECEIPT_PATH = ADAPTED_RECEIPT_PATH
    merge.CHECKPOINT_ROOT = raw_root / "checkpoints_v1_19_residual_direct_text_only"
    merge.OUTPUT_REQUIREMENT_PATH = (
        audit_root / "official_fact_requirement_ledger_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_QUEUE_PATH = (
        audit_root / "official_fact_document_queue_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_PATCH_MANIFEST_PATH = (
        audit_root
        / "official_fact_checkpoint_patch_manifest_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_RESIDUAL_PRIORITY_PATH = (
        audit_root / "official_fact_residual_priority_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_FACTS_PATH = (
        raw_root / "official_financial_facts_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_DEPENDENCY_PATH = (
        raw_root / "event_fact_dependency_ledger_v1_19_direct_text_only.parquet"
    )
    merge.OUTPUT_RECEIPT_PATH = raw_root / "receipt_v1_19_direct_text_only.json"
    merge.OUTPUT_REPORT_PATH = (
        ROOT
        / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
        "OFFICIAL_FACTS_V1_19_DIRECT_TEXT_ONLY.md"
    )
    return merge.main()


if __name__ == "__main__":
    raise SystemExit(main())
