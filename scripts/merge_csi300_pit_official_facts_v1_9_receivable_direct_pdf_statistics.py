from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts import merge_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as merge  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_V1_9_RECEIVABLE_DIRECT_PDF_MERGE_V1"
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_9_RECEIVABLE_DIRECT_PDF_MERGED_V1_0_0"
)


def main() -> int:
    audit_root = merge.AUDIT_ROOT
    raw_root = merge.RAW_ROOT
    merge.PROTOCOL_ID = PROTOCOL_ID
    merge.MERGED_PARSER_VERSION = MERGED_PARSER_VERSION
    merge.BASE_RECEIPT_PATH = raw_root / "receipt_v1_8_balance_direct_pdf.json"
    merge.RESIDUAL_RECEIPT_PATH = (
        ROOT / "tmp/pit_v1_9_receivable_residual_direct_pdf_statistics/receipt.json"
    )
    merge.CHECKPOINT_ROOT = (
        raw_root / "checkpoints_v1_9_receivable_direct_pdf_merged"
    )
    merge.OUTPUT_REQUIREMENT_PATH = (
        audit_root
        / "official_fact_requirement_ledger_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_QUEUE_PATH = (
        audit_root / "official_fact_document_queue_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_PATCH_MANIFEST_PATH = (
        audit_root
        / "official_fact_checkpoint_patch_manifest_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_RESIDUAL_PRIORITY_PATH = (
        audit_root
        / "official_fact_residual_priority_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_FACTS_PATH = (
        raw_root / "official_financial_facts_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_DEPENDENCY_PATH = (
        raw_root / "event_fact_dependency_ledger_v1_9_receivable_direct_pdf.parquet"
    )
    merge.OUTPUT_RECEIPT_PATH = raw_root / "receipt_v1_9_receivable_direct_pdf.json"
    merge.OUTPUT_REPORT_PATH = (
        ROOT
        / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_"
        "OFFICIAL_FACTS_V1_9_RECEIVABLE_DIRECT_PDF.md"
    )
    return merge.main()


if __name__ == "__main__":
    raise SystemExit(main())
