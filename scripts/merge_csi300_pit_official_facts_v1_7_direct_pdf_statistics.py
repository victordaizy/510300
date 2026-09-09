from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402
from scripts import collect_csi300_pit_fundamental_underreaction_official_facts_v1 as collector  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_V1_7_DIRECT_PDF_MERGE_V1"
MERGED_PARSER_VERSION = (
    "CSI300_PIT_OFFICIAL_FACTS_V1_7_DIRECT_PDF_AND_TARGETED_OCR_MERGED_V1_0_0"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")

BASE_CONFIG_PATH = (
    ROOT / "config/csi300_pit_fundamental_underreaction_official_facts_v1_6.yaml"
)
BASE_RECEIPT_PATH = (
    ROOT
    / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/receipt_v1_6.json"
)
ALL_GAP_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_all_gap_census/receipt.json"
COMPANION_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_companions/receipt.json"
DIRECT_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_statistics/receipt.json"
OCR_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_targeted_direct_pdf_ocr/receipt.json"

AUDIT_ROOT = (
    ROOT / "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1"
)
RAW_ROOT = ROOT / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1"
CHECKPOINT_ROOT = RAW_ROOT / "checkpoints_v1_7_direct_pdf_merged"
OUTPUT_REQUIREMENT_PATH = (
    AUDIT_ROOT / "official_fact_requirement_ledger_v1_7_direct_pdf.parquet"
)
OUTPUT_QUEUE_PATH = AUDIT_ROOT / "official_fact_document_queue_v1_7_direct_pdf.parquet"
OUTPUT_PATCH_MANIFEST_PATH = (
    AUDIT_ROOT / "official_fact_checkpoint_patch_manifest_v1_7_direct_pdf.parquet"
)
OUTPUT_RESIDUAL_PRIORITY_PATH = (
    AUDIT_ROOT / "official_fact_residual_priority_v1_7_direct_pdf.parquet"
)
OUTPUT_FACTS_PATH = RAW_ROOT / "official_financial_facts_v1_7_direct_pdf.parquet"
OUTPUT_DEPENDENCY_PATH = (
    RAW_ROOT / "event_fact_dependency_ledger_v1_7_direct_pdf.parquet"
)
OUTPUT_RECEIPT_PATH = RAW_ROOT / "receipt_v1_7_direct_pdf.json"
OUTPUT_REPORT_PATH = (
    ROOT
    / "reports/data_quality/CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_7_DIRECT_PDF.md"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _relative(path: Path) -> str:
    absolute_path = path.absolute()
    absolute_root = ROOT.absolute()
    try:
        return absolute_path.relative_to(absolute_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"制品路径不在工作区逻辑根目录内：{absolute_path}"
        ) from error


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_gzip_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    compressed = gzip.compress(serialized, compresslevel=9, mtime=0)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _assert_no_return_reads(label: str, receipt: dict[str, Any]) -> None:
    checks = {
        "market_price_read": receipt.get("market_price_read") is False,
        "future_return_read": receipt.get("future_return_read") is False,
        "return_evaluation": receipt.get("return_evaluation") == "NOT_ALLOWED",
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{label}存在收益读取或状态漂移：{failed}")


def _verify_base_artifact(
    base_receipt: dict[str, Any],
    artifact_key: str,
) -> Path:
    artifacts = base_receipt["artifacts"]
    path = ROOT / artifacts[f"{artifact_key}_path"]
    expected = str(artifacts[f"{artifact_key}_sha256"])
    observed = _sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"V1.6基线制品哈希漂移：{artifact_key}|{expected}|{observed}"
        )
    return path


def _metric_ids(metrics: Iterable[dict[str, Any]]) -> list[str]:
    return [str(metric["metric_id"]) for metric in metrics]


def _source_record(
    *,
    source_kind: str,
    announcement_id: str,
    official_pdf_url: str,
    official_pdf_sha256: str,
    official_pdf_size_bytes: int | None,
    pdf_page_count: int | None,
    path: str | None,
) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "official_pdf_announcement_id": str(announcement_id),
        "official_pdf_url": str(official_pdf_url),
        "official_pdf_sha256": str(official_pdf_sha256),
        "official_pdf_size_bytes": (
            int(official_pdf_size_bytes)
            if official_pdf_size_bytes is not None
            else None
        ),
        "pdf_page_count": int(pdf_page_count) if pdf_page_count is not None else None,
        "path": str(path) if path else None,
    }


def _pdf_page_count(path_value: str) -> int:
    path = ROOT / path_value
    content = path.read_bytes()
    document = parser._base.pdfium.PdfDocument(content)
    try:
        return len(document)
    finally:
        document.close()


def _all_gap_source(row: dict[str, Any]) -> dict[str, Any]:
    return _source_record(
        source_kind="CHECKPOINT_OFFICIAL_ORIGINAL_PDF",
        announcement_id=str(row["announcement_id"]),
        official_pdf_url=str(row["official_pdf_url"]),
        official_pdf_sha256=str(row["observed_pdf_sha256"]),
        official_pdf_size_bytes=int(row["observed_pdf_size_bytes"]),
        pdf_page_count=int(row["pdf_page_count"]),
        path=str(row["path"]),
    )


def _companion_source(row: dict[str, Any]) -> dict[str, Any]:
    selected_id = str(row.get("selected_companion_announcement_id") or "")
    candidate = next(
        (
            item
            for item in row.get("candidate_results") or []
            if str(item.get("companion_announcement_id") or "") == selected_id
        ),
        None,
    )
    if candidate is None:
        raise RuntimeError(
            f"选定同日官方全文来源不存在：{row['source_announcement_id']}"
        )
    path = str(candidate["path"])
    return _source_record(
        source_kind="SAME_DAY_OFFICIAL_FULL_TEXT_COMPANION",
        announcement_id=selected_id,
        official_pdf_url=str(candidate["companion_official_pdf_url"]),
        official_pdf_sha256=str(candidate["observed_pdf_sha256"]),
        official_pdf_size_bytes=int(candidate["observed_pdf_size_bytes"]),
        pdf_page_count=_pdf_page_count(path),
        path=path,
    )


def _repair_metric_sources(
    source_results: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source in source_results:
        if source.get("error"):
            continue
        admitted = list(
            (source.get("repair_receipt") or {}).get("admitted_metric_ids") or []
        )
        if not admitted:
            continue
        record = _source_record(
            source_kind=str(source.get("source_kind") or "OFFICIAL_PDF"),
            announcement_id=str(source.get("announcement_id") or ""),
            official_pdf_url=str(source.get("official_pdf_url") or ""),
            official_pdf_sha256=str(
                source.get("observed_pdf_sha256")
                or source.get("expected_pdf_sha256")
                or ""
            ),
            official_pdf_size_bytes=(
                int(source["observed_pdf_size_bytes"])
                if source.get("observed_pdf_size_bytes") is not None
                else None
            ),
            pdf_page_count=(
                int(source["pdf_page_count"])
                if source.get("pdf_page_count") is not None
                else None
            ),
            path=str(source.get("path") or ""),
        )
        for metric_id in admitted:
            if metric_id in result:
                raise RuntimeError(f"指标来源重复：{metric_id}")
            result[str(metric_id)] = record
    return result


def _put_phase_metrics(
    result: dict[str, Any],
    metrics: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    *,
    phase: str,
    metric_source_documents: dict[str, dict[str, Any]],
) -> list[str]:
    admitted: list[str] = []
    for metric in metrics:
        metric_id = str(metric["metric_id"])
        if metric_id not in result.get("missing_metrics", []):
            raise RuntimeError(f"阶段指标不是缺失项：{phase}|{metric_id}")
        source = sources.get(metric_id)
        if source is None:
            raise RuntimeError(f"阶段指标缺少官方PDF来源：{phase}|{metric_id}")
        parser._v1_4._put_metric(
            result,
            parser._base.MetricEvidence(**metric),
        )
        metric_source_documents[metric_id] = {**source, "admission_phase": phase}
        admitted.append(metric_id)
    return admitted


def _checkpoint_output_path(result: dict[str, Any]) -> Path:
    year = str(result["report_period"])[:4]
    return CHECKPOINT_ROOT / year / f"{result['announcement_id']}.json.gz"


def _merge_checkpoint(
    direct_row: dict[str, Any],
    ocr_row: dict[str, Any] | None,
    all_gap_row: dict[str, Any],
    companion_row: dict[str, Any] | None,
    *,
    receipt_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_checkpoint_path = ROOT / direct_row["checkpoint_path"]
    source_checkpoint_content = source_checkpoint_path.read_bytes()
    source_checkpoint_sha256 = _sha256(source_checkpoint_content)
    if source_checkpoint_sha256 != str(direct_row["checkpoint_sha256"]):
        raise RuntimeError(f"来源检查点哈希漂移：{direct_row['announcement_id']}")
    result = _load_json_gzip(source_checkpoint_path)
    if list(result.get("missing_metrics") or []) != list(
        all_gap_row.get("before_missing_metrics") or []
    ):
        raise RuntimeError(f"V1.6缺失基线不一致：{direct_row['announcement_id']}")

    metric_source_documents: dict[str, dict[str, Any]] = {}
    phase_counts: dict[str, int] = {}

    all_gap_metrics = list(direct_row.get("all_gap_new_metrics") or [])
    all_gap_source = _all_gap_source(all_gap_row)
    phase_counts["V1_7_ALL_GAP_PARSER"] = len(
        _put_phase_metrics(
            result,
            all_gap_metrics,
            {metric_id: all_gap_source for metric_id in _metric_ids(all_gap_metrics)},
            phase="V1_7_ALL_GAP_PARSER",
            metric_source_documents=metric_source_documents,
        )
    )

    companion_metrics = list(direct_row.get("companion_new_metrics") or [])
    if companion_metrics:
        if companion_row is None:
            raise RuntimeError(f"同日官方全文收据缺失：{direct_row['announcement_id']}")
        companion_source = _companion_source(companion_row)
        companion_sources = {
            metric_id: companion_source for metric_id in _metric_ids(companion_metrics)
        }
    else:
        companion_sources = {}
    phase_counts["SAME_DAY_OFFICIAL_FULL_TEXT_COMPANION"] = len(
        _put_phase_metrics(
            result,
            companion_metrics,
            companion_sources,
            phase="SAME_DAY_OFFICIAL_FULL_TEXT_COMPANION",
            metric_source_documents=metric_source_documents,
        )
    )

    direct_metrics = list(direct_row.get("new_metrics") or [])
    phase_counts["DIRECT_PDF_DOCUMENT_STATISTICS"] = len(
        _put_phase_metrics(
            result,
            direct_metrics,
            _repair_metric_sources(direct_row.get("source_results") or []),
            phase="DIRECT_PDF_DOCUMENT_STATISTICS",
            metric_source_documents=metric_source_documents,
        )
    )
    if list(result.get("missing_metrics") or []) != list(
        direct_row.get("after_missing_metrics") or []
    ):
        raise RuntimeError(f"逐份PDF直接统计复算不一致：{direct_row['announcement_id']}")

    ocr_metrics = list((ocr_row or {}).get("new_metrics") or [])
    phase_counts["TARGETED_DUAL_SCALE_OCR"] = len(
        _put_phase_metrics(
            result,
            ocr_metrics,
            _repair_metric_sources((ocr_row or {}).get("source_results") or []),
            phase="TARGETED_DUAL_SCALE_OCR",
            metric_source_documents=metric_source_documents,
        )
    )
    expected_final_missing = list(
        (
            (ocr_row or {}).get("after_missing_metrics")
            if ocr_row is not None
            else direct_row.get("after_missing_metrics")
        )
        or []
    )
    if list(result.get("missing_metrics") or []) != expected_final_missing:
        raise RuntimeError(f"定向OCR最终复算不一致：{direct_row['announcement_id']}")

    metric_ids = _metric_ids(result.get("metrics") or [])
    if len(metric_ids) != len(set(metric_ids)):
        raise RuntimeError(f"合并检查点公告-指标键重复：{direct_row['announcement_id']}")
    if len(metric_ids) + len(expected_final_missing) != len(parser.REQUIRED_METRICS):
        raise RuntimeError(f"合并检查点九指标守恒失败：{direct_row['announcement_id']}")

    source_status = str(result.get("checkpoint_status") or "PARSED_INCOMPLETE")
    final_complete = not expected_final_missing
    result["pre_merge_checkpoint_status"] = source_status
    result["pre_merge_parser_version"] = str(result.get("parser_version") or "")
    result["parser_version"] = MERGED_PARSER_VERSION
    result["checkpoint_status"] = (
        "PARSED_COMPLETE" if final_complete else "PARSED_INCOMPLETE"
    )
    result["document_terminal"] = True
    result["document_complete"] = final_complete
    result["document_status"] = (
        "PASS_ALL_REQUIRED_OFFICIAL_FINANCIAL_FACTS_EXTRACTED"
        if final_complete
        else "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE"
    )
    result["missing_metrics"] = expected_final_missing
    result["metric_source_documents"] = metric_source_documents
    result["direct_pdf_statistics_merge_receipt"] = {
        "protocol_id": PROTOCOL_ID,
        "source_checkpoint_path": _relative(source_checkpoint_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "input_receipt_sha256": receipt_hashes,
        "phase_new_metric_counts": phase_counts,
        "total_new_metric_count": sum(phase_counts.values()),
        "final_metric_count": len(metric_ids),
        "final_missing_metrics": expected_final_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    result["market_price_read"] = False
    result["future_return_read"] = False
    result["future_label_created"] = False
    result["portfolio_return_calculated"] = False
    result["return_evaluation"] = "NOT_ALLOWED"

    output_path = _checkpoint_output_path(result)
    _atomic_write_gzip_json(output_path, result)
    output_sha256 = _sha256_file(output_path)
    manifest_row = {
        "announcement_id": str(result["announcement_id"]),
        "ts_code": str(result["ts_code"]),
        "report_period": str(result["report_period"]),
        "period_type": str(result["period_type"]),
        "source_checkpoint_path": _relative(source_checkpoint_path),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "merged_checkpoint_path": _relative(output_path),
        "merged_checkpoint_sha256": output_sha256,
        "base_metric_count": len(metric_ids) - sum(phase_counts.values()),
        "new_metric_count": sum(phase_counts.values()),
        "all_gap_new_metric_count": phase_counts["V1_7_ALL_GAP_PARSER"],
        "companion_new_metric_count": phase_counts[
            "SAME_DAY_OFFICIAL_FULL_TEXT_COMPANION"
        ],
        "direct_statistics_new_metric_count": phase_counts[
            "DIRECT_PDF_DOCUMENT_STATISTICS"
        ],
        "targeted_ocr_new_metric_count": phase_counts["TARGETED_DUAL_SCALE_OCR"],
        "final_metric_count": len(metric_ids),
        "final_missing_metric_count": len(expected_final_missing),
        "final_missing_metrics_json": json.dumps(
            expected_final_missing,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "document_complete": final_complete,
        "checkpoint_status": result["checkpoint_status"],
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    return result, manifest_row


def _fact_rows_from_checkpoint(
    checkpoint: dict[str, Any],
    queue_row: dict[str, Any],
) -> list[dict[str, Any]]:
    specific_sources = checkpoint.get("metric_source_documents") or {}
    default_source = {
        "official_pdf_url": str(checkpoint["official_pdf_url"]),
        "official_pdf_announcement_id": str(
            checkpoint["official_pdf_announcement_id"]
        ),
        "official_pdf_sha256": str(checkpoint["official_pdf_sha256"]),
        "official_pdf_size_bytes": int(checkpoint["official_pdf_size_bytes"]),
        "pdf_page_count": int(checkpoint["pdf_page_count"]),
    }
    rows: list[dict[str, Any]] = []
    for metric in checkpoint.get("metrics") or []:
        metric_id = str(metric["metric_id"])
        source = {**default_source, **specific_sources.get(metric_id, {})}
        if source.get("official_pdf_size_bytes") is None:
            raise RuntimeError(
                f"指标来源缺少PDF字节数：{checkpoint['announcement_id']}|{metric_id}"
            )
        if source.get("pdf_page_count") is None:
            path = source.get("path")
            if not path:
                raise RuntimeError(
                    f"指标来源缺少PDF页数：{checkpoint['announcement_id']}|{metric_id}"
                )
            source["pdf_page_count"] = _pdf_page_count(str(path))
        rows.append(
            {
                "announcement_id": str(queue_row["announcement_id"]),
                "ts_code": str(queue_row["ts_code"]),
                "report_period": collector.timestamp_iso(queue_row["report_period"]),
                "period_type": str(queue_row["period_type"]),
                "event_publication_date": collector.timestamp_iso(
                    queue_row["event_publication_date"]
                ),
                "official_timestamp_at": collector.utc_or_local_iso(
                    queue_row.get("official_timestamp_at")
                ),
                "official_pdf_url": str(source["official_pdf_url"]),
                "official_pdf_announcement_id": str(
                    source["official_pdf_announcement_id"]
                ),
                "official_pdf_sha256": str(source["official_pdf_sha256"]),
                "official_pdf_size_bytes": int(source["official_pdf_size_bytes"]),
                "pdf_page_count": int(source["pdf_page_count"]),
                "parser_version": str(checkpoint["parser_version"]),
                **metric,
                "market_price_read": False,
                "future_return_read": False,
            }
        )
    return rows


def _coverage_delta(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    key_columns: list[str],
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(column) or "") for column in key_columns)

    before_map = {key(row): row for row in before_rows}
    after_map = {key(row): row for row in after_rows}
    result: list[dict[str, Any]] = []
    for item_key in sorted(after_map):
        after = after_map[item_key]
        before = before_map[item_key]
        result.append(
            {
                **{column: after[column] for column in key_columns},
                "target_event_count": int(after["target_event_count"]),
                "complete_event_count_before": int(before["complete_event_count"]),
                "complete_event_count_after": int(after["complete_event_count"]),
                "newly_ready_event_count": int(
                    after["complete_event_count"] - before["complete_event_count"]
                ),
                "complete_event_ratio_before": float(before["complete_event_ratio"]),
                "complete_event_ratio_after": float(after["complete_event_ratio"]),
                "complete_event_ratio_delta": float(
                    after["complete_event_ratio"] - before["complete_event_ratio"]
                ),
            }
        )
    return result


def _admission_checks(
    coverage: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    admission = config["admission"]
    overall_threshold = float(admission["required_complete_event_ratio"])
    year_threshold = float(
        admission["required_complete_event_ratio_each_publication_year"]
    )
    industry_threshold = float(
        admission[
            "required_complete_event_ratio_each_industry_with_at_least_50_targets"
        ]
    )
    failed_years = [
        row
        for row in coverage["publication_year_coverage"]
        if float(row["complete_event_ratio"]) < year_threshold
    ]
    failed_industries = [
        row
        for row in coverage["industry_coverage"]
        if int(row["target_event_count"]) >= 50
        and float(row["complete_event_ratio"]) < industry_threshold
    ]
    return {
        "overall": {
            "threshold": overall_threshold,
            "observed": float(coverage["overall_complete_event_ratio"]),
            "passed": float(coverage["overall_complete_event_ratio"])
            >= overall_threshold,
        },
        "each_publication_year": {
            "threshold": year_threshold,
            "minimum_observed": float(
                coverage["minimum_publication_year_complete_event_ratio"]
            ),
            "passed": not failed_years,
            "failed_rows": failed_years,
        },
        "each_eligible_industry": {
            "minimum_target_count": 50,
            "threshold": industry_threshold,
            "minimum_observed": float(
                coverage["minimum_eligible_industry_complete_event_ratio"]
            ),
            "passed": not failed_industries,
            "failed_rows": failed_industries,
        },
        "all_passed": bool(coverage["coverage_gate_passed"]),
    }


def _residual_priority(
    runtime_requirement: pd.DataFrame,
    queue_state: pd.DataFrame,
) -> pd.DataFrame:
    direct_unlock_counts: Counter[str] = Counter()
    for _, group in runtime_requirement.groupby("target_announcement_id", sort=False):
        missing_metadata = bool(
            ((~group["queued_for_official_pdf"]) | (~group["document_terminal"])).any()
        )
        incomplete_ids = sorted(
            {
                str(value)
                for value in group.loc[
                    group["queued_for_official_pdf"]
                    & group["document_terminal"]
                    & ~group["document_complete"],
                    "dependency_announcement_id",
                ].dropna()
            }
        )
        if not missing_metadata and len(incomplete_ids) == 1:
            direct_unlock_counts[incomplete_ids[0]] += 1

    residual = queue_state.loc[~queue_state["document_complete"]].copy()
    residual["single_document_unlock_event_count"] = residual["announcement_id"].map(
        lambda value: int(direct_unlock_counts.get(str(value), 0))
    )
    selected_columns = [
        "announcement_id",
        "ts_code",
        "report_period",
        "period_type",
        "event_publication_date",
        "dependent_target_event_count",
        "single_document_unlock_event_count",
        "missing_metric_count",
        "missing_metrics_json",
        "checkpoint_path",
        "checkpoint_sha256",
    ]
    residual = residual[selected_columns].sort_values(
        [
            "single_document_unlock_event_count",
            "dependent_target_event_count",
            "missing_metric_count",
            "event_publication_date",
            "announcement_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    residual["market_price_read"] = False
    residual["future_return_read"] = False
    residual["return_evaluation"] = "NOT_ALLOWED"
    return residual.reset_index(drop=True)


def _render_report(receipt: dict[str, Any], residual: pd.DataFrame) -> str:
    counts = receipt["counts"]
    coverage = receipt["coverage"]
    checks = receipt["admission_checks"]
    counterfactual = receipt["counterfactual_all_remaining_documents_complete"]
    failed_years = checks["each_publication_year"]["failed_rows"]
    failed_industries = checks["each_eligible_industry"]["failed_rows"]
    lines = [
        "# 沪深300基本面反应不足：官方PDF逐份统计与覆盖复算",
        "",
        f"- 状态：`{receipt['status']}`",
        f"- 官方PDF文档：{counts['terminal_document_count']:,}/{counts['queued_document_count']:,} 已终局",
        f"- 完整文档：{counts['complete_document_count']:,}；仍不完整：{counts['incomplete_terminal_document_count']:,}",
        f"- 财务事实：{counts['fact_row_count']:,}；本轮新增：{counts['new_fact_row_count']:,}",
        f"- 目标事件可用：{counts['ready_target_event_count']:,}/{counts['target_event_count']:,}",
        f"- 本轮新增可用事件：{counts['newly_ready_target_event_count']:,}",
        "",
        "## 冻结覆盖门槛",
        "",
        f"- 总体：{coverage['overall_complete_event_ratio']:.6%}，门槛 {checks['overall']['threshold']:.0%}，通过={checks['overall']['passed']}",
        f"- 最低年度：{coverage['minimum_publication_year_complete_event_ratio']:.6%}，门槛 {checks['each_publication_year']['threshold']:.0%}，通过={checks['each_publication_year']['passed']}",
        f"- 最低合格行业：{coverage['minimum_eligible_industry_complete_event_ratio']:.6%}，门槛 {checks['each_eligible_industry']['threshold']:.0%}，通过={checks['each_eligible_industry']['passed']}",
        "",
        "## 未通过项",
        "",
        "### 年度",
        "",
        "| 年度 | 完整事件 | 目标事件 | 覆盖率 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {int(row['publication_year'])} | {int(row['complete_event_count'])} | {int(row['target_event_count'])} | {float(row['complete_event_ratio']):.4%} |"
        for row in failed_years
    )
    lines.extend(
        [
            "",
            "### 合格行业",
            "",
            "| 行业 | 完整事件 | 目标事件 | 覆盖率 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {row['industry_l1']} | {int(row['complete_event_count'])} | {int(row['target_event_count'])} | {float(row['complete_event_ratio']):.4%} |"
        for row in failed_industries
    )
    lines.extend(
        [
            "",
            "## 剩余缺口的上界",
            "",
            f"若其余 {counts['incomplete_terminal_document_count']:,} 份全部补齐，可用事件为 {counterfactual['ready_target_event_count']:,}/{counts['target_event_count']:,}，总体覆盖率 {counterfactual['coverage']['overall_complete_event_ratio']:.6%}，三层门槛全部通过={counterfactual['coverage']['coverage_gate_passed']}。",
            "",
            "## 下一轮优先文档",
            "",
            "| 公告编号 | 证券 | 期别 | 缺失数 | 单份补齐可直接解锁事件 | 被依赖事件数 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in residual.head(20).to_dict("records"):
        lines.append(
            f"| {row['announcement_id']} | {row['ts_code']} | {row['period_type']} | "
            f"{int(row['missing_metric_count'])} | "
            f"{int(row['single_document_unlock_event_count'])} | "
            f"{int(row['dependent_target_event_count'])} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本轮只合并官方PDF财务事实并复算覆盖率；未读取市场价格、未来收益，未生成标签、信号、组合、持仓或订单。覆盖门槛未全部通过时，收益评价仍为 `NOT_ALLOWED`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    base_receipt = _load_json(BASE_RECEIPT_PATH)
    all_gap_receipt = _load_json(ALL_GAP_RECEIPT_PATH)
    companion_receipt = _load_json(COMPANION_RECEIPT_PATH)
    direct_receipt = _load_json(DIRECT_RECEIPT_PATH)
    ocr_receipt = _load_json(OCR_RECEIPT_PATH)

    if base_receipt.get("market_price_read") is not False or base_receipt.get(
        "future_return_read"
    ) is not False:
        raise RuntimeError("V1.6基线读取状态不满足无收益门槛")
    for label, source in (
        ("V1.7全缺口收据", all_gap_receipt),
        ("同日官方全文收据", companion_receipt),
        ("逐份PDF直接统计收据", direct_receipt),
        ("定向OCR收据", ocr_receipt),
    ):
        _assert_no_return_reads(label, source)
    if int(all_gap_receipt.get("exception_count", -1)) != 0:
        raise RuntimeError("V1.7全缺口收据存在异常")
    if int(companion_receipt.get("exception_count", -1)) != 0:
        raise RuntimeError("同日官方全文收据存在异常")
    if int(direct_receipt.get("exception_count", -1)) != 0:
        raise RuntimeError("逐份PDF直接统计收据存在异常")
    if int(ocr_receipt.get("target_exception_count", -1)) != 0 or int(
        ocr_receipt.get("source_exception_count", -1)
    ) != 0:
        raise RuntimeError("定向OCR收据存在异常")

    receipt_hashes = {
        "base_v1_6": _sha256_file(BASE_RECEIPT_PATH),
        "all_gap_v1_7": _sha256_file(ALL_GAP_RECEIPT_PATH),
        "companion_v1_7": _sha256_file(COMPANION_RECEIPT_PATH),
        "direct_pdf_statistics": _sha256_file(DIRECT_RECEIPT_PATH),
        "targeted_direct_pdf_ocr": _sha256_file(OCR_RECEIPT_PATH),
    }
    if str(ocr_receipt["input_direct_receipt_sha256"]) != receipt_hashes[
        "direct_pdf_statistics"
    ]:
        raise RuntimeError("定向OCR输入的逐份PDF收据哈希不一致")

    base_requirement_path = _verify_base_artifact(
        base_receipt,
        "requirement_ledger",
    )
    base_queue_path = _verify_base_artifact(base_receipt, "document_queue")
    base_facts_path = _verify_base_artifact(base_receipt, "facts")
    base_dependency_path = _verify_base_artifact(base_receipt, "dependency_ledger")

    base_requirement_runtime = pd.read_parquet(base_requirement_path)
    base_queue = pd.read_parquet(base_queue_path)
    base_facts = pd.read_parquet(base_facts_path)
    base_dependency = pd.read_parquet(base_dependency_path)

    direct_rows = list(direct_receipt.get("results") or [])
    direct_map = {str(row["announcement_id"]): row for row in direct_rows}
    all_gap_map = {
        str(row["announcement_id"]): row
        for row in all_gap_receipt.get("results") or []
    }
    companion_map = {
        str(row["source_announcement_id"]): row
        for row in companion_receipt.get("results") or []
    }
    ocr_map = {
        str(row["announcement_id"]): row
        for row in ocr_receipt.get("results") or []
    }
    base_incomplete_ids = set(
        base_queue.loc[
            ~base_queue["document_complete"],
            "announcement_id",
        ].astype(str)
    )
    if set(direct_map) != base_incomplete_ids or set(all_gap_map) != base_incomplete_ids:
        raise RuntimeError("373份V1.6缺口与逐份统计输入集合不一致")
    if not set(ocr_map).issubset(base_incomplete_ids):
        raise RuntimeError("定向OCR目标超出V1.6缺口集合")

    merged_checkpoints: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for index, announcement_id in enumerate(sorted(base_incomplete_ids), start=1):
        checkpoint, manifest_row = _merge_checkpoint(
            direct_map[announcement_id],
            ocr_map.get(announcement_id),
            all_gap_map[announcement_id],
            companion_map.get(announcement_id),
            receipt_hashes=receipt_hashes,
        )
        merged_checkpoints[announcement_id] = checkpoint
        manifest_rows.append(manifest_row)
        if index % 50 == 0 or index == len(base_incomplete_ids):
            print(f"合并检查点 {index}/{len(base_incomplete_ids)}", flush=True)

    patch_manifest = pd.DataFrame(manifest_rows).sort_values(
        ["report_period", "ts_code", "announcement_id"],
        kind="stable",
    )
    _atomic_write_parquet(OUTPUT_PATCH_MANIFEST_PATH, patch_manifest)

    queue_state = base_queue.copy()
    queue_state["statistics_merge_protocol_id"] = None
    queue_state["statistics_new_metric_count"] = 0
    queue_state["statistics_source_checkpoint_sha256"] = None
    manifest_map = {
        str(row["announcement_id"]): row for row in patch_manifest.to_dict("records")
    }
    for announcement_id, manifest_row in manifest_map.items():
        mask = queue_state["announcement_id"].astype(str).eq(announcement_id)
        if int(mask.sum()) != 1:
            raise RuntimeError(f"文档队列公告键不唯一：{announcement_id}")
        checkpoint = merged_checkpoints[announcement_id]
        queue_state.loc[mask, "checkpoint_path"] = manifest_row[
            "merged_checkpoint_path"
        ]
        queue_state.loc[mask, "checkpoint_sha256"] = manifest_row[
            "merged_checkpoint_sha256"
        ]
        queue_state.loc[mask, "checkpoint_status"] = checkpoint["checkpoint_status"]
        queue_state.loc[mask, "document_terminal"] = True
        queue_state.loc[mask, "document_complete"] = bool(
            checkpoint["document_complete"]
        )
        queue_state.loc[mask, "document_status"] = checkpoint["document_status"]
        queue_state.loc[mask, "missing_metric_count"] = len(
            checkpoint["missing_metrics"]
        )
        queue_state.loc[mask, "missing_metrics_json"] = json.dumps(
            checkpoint["missing_metrics"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        queue_state.loc[mask, "statistics_merge_protocol_id"] = PROTOCOL_ID
        queue_state.loc[mask, "statistics_new_metric_count"] = int(
            manifest_row["new_metric_count"]
        )
        queue_state.loc[mask, "statistics_source_checkpoint_sha256"] = manifest_row[
            "source_checkpoint_sha256"
        ]
    _atomic_write_parquet(OUTPUT_QUEUE_PATH, queue_state)

    patch_ids = set(merged_checkpoints)
    rebuilt_fact_rows: list[dict[str, Any]] = []
    queue_map = {
        str(row["announcement_id"]): row for row in queue_state.to_dict("records")
    }
    for announcement_id in sorted(patch_ids):
        rebuilt_fact_rows.extend(
            _fact_rows_from_checkpoint(
                merged_checkpoints[announcement_id],
                queue_map[announcement_id],
            )
        )
    rebuilt_facts = pd.DataFrame(rebuilt_fact_rows, columns=collector.FACT_COLUMNS)
    facts = pd.concat(
        [
            base_facts.loc[
                ~base_facts["announcement_id"].astype(str).isin(patch_ids)
            ],
            rebuilt_facts,
        ],
        ignore_index=True,
    )
    if facts.duplicated(["announcement_id", "metric_id"]).any():
        raise RuntimeError("合并事实公告-指标键不唯一")
    if not facts["metric_value_cny"].map(math.isfinite).all():
        raise RuntimeError("合并事实存在非有限数值")
    if facts["market_price_read"].any() or facts["future_return_read"].any():
        raise RuntimeError("合并事实读取状态漂移")
    facts = facts.sort_values(
        ["report_period", "ts_code", "announcement_id", "metric_id"],
        kind="stable",
    ).reset_index(drop=True)
    observed_metric_counts = facts.groupby("announcement_id")["metric_id"].size()
    expected_metric_counts = queue_state.set_index("announcement_id")[
        "missing_metric_count"
    ].map(lambda value: len(parser.REQUIRED_METRICS) - int(value))
    if not observed_metric_counts.reindex(expected_metric_counts.index).fillna(0).astype(
        int
    ).eq(expected_metric_counts.astype(int)).all():
        raise RuntimeError("合并事实数量与文档缺失计数不守恒")
    _atomic_write_parquet(OUTPUT_FACTS_PATH, facts)

    state_columns = [
        "checkpoint_status",
        "document_terminal",
        "document_complete",
        "missing_metric_count",
        "missing_metrics_json",
        "requirement_status",
    ]
    base_requirement = base_requirement_runtime.drop(
        columns=[
            column
            for column in state_columns
            if column in base_requirement_runtime.columns
        ]
    )
    runtime_requirement, dependency = collector.build_event_dependency_ledger(
        base_requirement,
        queue_state,
    )
    coverage = collector.coverage_metrics(dependency, config)
    _atomic_write_parquet(OUTPUT_REQUIREMENT_PATH, runtime_requirement)
    _atomic_write_parquet(OUTPUT_DEPENDENCY_PATH, dependency)

    residual_priority = _residual_priority(runtime_requirement, queue_state)
    _atomic_write_parquet(OUTPUT_RESIDUAL_PRIORITY_PATH, residual_priority)

    counterfactual_queue = queue_state.copy()
    incomplete_mask = ~counterfactual_queue["document_complete"]
    counterfactual_queue.loc[incomplete_mask, "checkpoint_status"] = "PARSED_COMPLETE"
    counterfactual_queue.loc[incomplete_mask, "document_complete"] = True
    counterfactual_queue.loc[incomplete_mask, "document_terminal"] = True
    counterfactual_queue.loc[incomplete_mask, "missing_metric_count"] = 0
    counterfactual_queue.loc[incomplete_mask, "missing_metrics_json"] = "[]"
    _, counterfactual_dependency = collector.build_event_dependency_ledger(
        base_requirement,
        counterfactual_queue,
    )
    counterfactual_coverage = collector.coverage_metrics(
        counterfactual_dependency,
        config,
    )

    base_coverage = base_receipt["coverage"]
    year_delta = _coverage_delta(
        base_coverage["publication_year_coverage"],
        coverage["publication_year_coverage"],
        ["publication_year"],
    )
    industry_delta = _coverage_delta(
        base_coverage["industry_coverage"],
        coverage["industry_coverage"],
        ["industry_l1_code", "industry_l1"],
    )
    admission_checks = _admission_checks(coverage, config)
    ready_count = int(dependency["target_event_ready"].sum())
    base_ready_count = int(base_receipt["counts"]["ready_target_event_count"])
    complete_count = int(queue_state["document_complete"].sum())
    terminal_count = int(queue_state["document_terminal"].sum())
    status = (
        str(config["admission"]["pass_status"])
        if coverage["coverage_gate_passed"]
        else str(config["admission"]["coverage_fail_status"])
    )
    counts = {
        "target_event_count": int(len(dependency)),
        "requirement_count": int(len(runtime_requirement)),
        "temporally_available_requirement_count": int(
            runtime_requirement["queued_for_official_pdf"].sum()
        ),
        "queued_document_count": int(len(queue_state)),
        "terminal_document_count": terminal_count,
        "complete_document_count": complete_count,
        "incomplete_terminal_document_count": int(
            terminal_count - complete_count
        ),
        "nonterminal_document_count": int(len(queue_state) - terminal_count),
        "download_failed_document_count": int(
            queue_state["checkpoint_status"].eq("DOWNLOAD_FAILED").sum()
        ),
        "fact_row_count": int(len(facts)),
        "base_fact_row_count": int(len(base_facts)),
        "new_fact_row_count": int(len(facts) - len(base_facts)),
        "ready_target_event_count": ready_count,
        "base_ready_target_event_count": base_ready_count,
        "newly_ready_target_event_count": ready_count - base_ready_count,
    }
    residual_missing_metric_counts = Counter(
        metric_id
        for checkpoint in merged_checkpoints.values()
        for metric_id in checkpoint.get("missing_metrics") or []
    )
    receipt: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": "1.0.0",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "status": status,
        "parser_version": MERGED_PARSER_VERSION,
        "counts": counts,
        "checkpoint_status_counts": queue_state["checkpoint_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "requirement_status_counts": runtime_requirement["requirement_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "dependency_status_counts": dependency["dependency_status"]
        .value_counts(dropna=False)
        .to_dict(),
        "coverage": coverage,
        "admission_checks": admission_checks,
        "coverage_change_from_v1_6": {
            "overall_complete_event_ratio_before": float(
                base_coverage["overall_complete_event_ratio"]
            ),
            "overall_complete_event_ratio_after": float(
                coverage["overall_complete_event_ratio"]
            ),
            "overall_complete_event_ratio_delta": float(
                coverage["overall_complete_event_ratio"]
                - base_coverage["overall_complete_event_ratio"]
            ),
            "publication_year_rows": year_delta,
            "industry_rows": industry_delta,
        },
        "counterfactual_all_remaining_documents_complete": {
            "remaining_document_count": int((~queue_state["document_complete"]).sum()),
            "ready_target_event_count": int(
                counterfactual_dependency["target_event_ready"].sum()
            ),
            "coverage": counterfactual_coverage,
        },
        "residual": {
            "document_count": int(len(residual_priority)),
            "missing_metric_counts": dict(
                sorted(
                    residual_missing_metric_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "single_document_unlockable_event_count": int(
                residual_priority["single_document_unlock_event_count"].sum()
            ),
            "document_with_positive_single_unlock_count": int(
                residual_priority["single_document_unlock_event_count"].gt(0).sum()
            ),
        },
        "provenance": {
            "base_config_path": _relative(BASE_CONFIG_PATH),
            "base_config_sha256": _sha256_file(BASE_CONFIG_PATH),
            "base_receipt_path": _relative(BASE_RECEIPT_PATH),
            "input_receipt_sha256": receipt_hashes,
            "base_artifact_sha256_verified": True,
            "checkpoint_patch_count": int(len(patch_manifest)),
        },
        "market_price_read": False,
        "future_return_read": False,
        "future_label_created": False,
        "signal_score_calculated": False,
        "portfolio_return_calculated": False,
        "positions_generated": False,
        "orders_generated": False,
        "broker_connection_performed": False,
        "trading_authorization": "NO_TRADE",
        "return_evaluation": "NOT_ALLOWED",
        "next_allowed_step": (
            "FREEZE_PHASE_1_ANALYSIS_PROTOCOL_BEFORE_ANY_RETURN_READ"
            if coverage["coverage_gate_passed"]
            else "CONTINUE_OFFICIAL_PDF_FACT_COMPLETION_WITH_RESIDUAL_PRIORITY"
        ),
    }
    report = _render_report(receipt, residual_priority)
    _atomic_write_text(OUTPUT_REPORT_PATH, report)
    receipt["artifacts"] = {
        "requirement_ledger_path": _relative(OUTPUT_REQUIREMENT_PATH),
        "requirement_ledger_sha256": _sha256_file(OUTPUT_REQUIREMENT_PATH),
        "document_queue_path": _relative(OUTPUT_QUEUE_PATH),
        "document_queue_sha256": _sha256_file(OUTPUT_QUEUE_PATH),
        "checkpoint_patch_manifest_path": _relative(OUTPUT_PATCH_MANIFEST_PATH),
        "checkpoint_patch_manifest_sha256": _sha256_file(
            OUTPUT_PATCH_MANIFEST_PATH
        ),
        "residual_priority_path": _relative(OUTPUT_RESIDUAL_PRIORITY_PATH),
        "residual_priority_sha256": _sha256_file(OUTPUT_RESIDUAL_PRIORITY_PATH),
        "facts_path": _relative(OUTPUT_FACTS_PATH),
        "facts_sha256": _sha256_file(OUTPUT_FACTS_PATH),
        "dependency_ledger_path": _relative(OUTPUT_DEPENDENCY_PATH),
        "dependency_ledger_sha256": _sha256_file(OUTPUT_DEPENDENCY_PATH),
        "report_markdown_path": _relative(OUTPUT_REPORT_PATH),
        "report_markdown_sha256": _sha256_file(OUTPUT_REPORT_PATH),
    }
    _atomic_write_json(OUTPUT_RECEIPT_PATH, receipt)

    print(
        "逐份PDF统计合并完成："
        f"完整文档={complete_count}/{len(queue_state)}；"
        f"事实={len(facts)}；"
        f"可用事件={ready_count}/{len(dependency)}；"
        f"总体覆盖={coverage['overall_complete_event_ratio']:.6%}；"
        f"覆盖门={coverage['coverage_gate_passed']}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
