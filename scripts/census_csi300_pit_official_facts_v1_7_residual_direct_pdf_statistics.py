from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_residual_v1 as residual  # noqa: E402
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402


DIRECT_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_statistics/receipt.json"
PATCH_MANIFEST_PATH = ROOT / (
    "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "official_fact_checkpoint_patch_manifest_v1_7_direct_pdf.parquet"
)
DEFAULT_OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_residual_direct_pdf_statistics"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_RESIDUAL_DIRECT_PDF_CENSUS_V1"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _source_contract(source: dict[str, Any]) -> dict[str, Any] | None:
    if source.get("error"):
        return None
    path = str(source.get("path") or "")
    expected_sha256 = str(
        source.get("observed_pdf_sha256")
        or source.get("expected_pdf_sha256")
        or ""
    )
    if not path or not expected_sha256:
        return None
    return {
        "source_kind": str(source.get("source_kind") or "OFFICIAL_PDF"),
        "announcement_id": str(source.get("announcement_id") or ""),
        "official_pdf_url": str(source.get("official_pdf_url") or ""),
        "path": path,
        "expected_pdf_sha256": expected_sha256,
    }


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(metric["metric_id"]): metric
        for metric in result.get("metrics") or []
        if metric.get("metric_id")
    }


def _summary_table_page_indices(page_texts: Sequence[str]) -> tuple[int, ...]:
    upper = min(40, len(page_texts))
    selected: set[int] = set()
    for page_index in range(upper):
        if not residual._is_summary_page(page_texts[page_index]):
            continue
        for adjacent in range(max(0, page_index - 1), min(upper, page_index + 2)):
            selected.add(adjacent)
    return tuple(sorted(selected))


def _extract_selected_tables(
    content: bytes,
    page_indices: Sequence[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    tables: dict[int, Any] = {}
    errors: list[dict[str, Any]] = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for page_index in page_indices:
            if page_index >= len(pdf.pages):
                continue
            try:
                extracted = pdf.pages[page_index].extract_tables() or []
                if extracted:
                    tables[page_index] = extracted
            except Exception as error:
                errors.append(
                    {
                        "page_number": page_index + 1,
                        "error": f"{type(error).__name__}: {error}"[:2000],
                    }
                )
    return tables, {
        "selected_page_numbers": [page_index + 1 for page_index in page_indices],
        "selected_page_count": len(page_indices),
        "page_with_table_count": len(tables),
        "table_count": sum(len(value) for value in tables.values()),
        "page_errors": errors,
    }


def _run_source(
    result: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    path = ROOT / source["path"]
    content = path.read_bytes()
    observed_sha256 = _sha256(content)
    if observed_sha256 != source["expected_pdf_sha256"]:
        raise RuntimeError(
            "官方PDF哈希漂移："
            f"{target['announcement_id']}|{source['announcement_id']}"
        )
    if not content.startswith(b"%PDF-"):
        raise RuntimeError(
            "官方附件没有PDF文件头："
            f"{target['announcement_id']}|{source['announcement_id']}"
        )
    page_texts, text_receipt = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(page_text) for page_text in page_texts
    ]
    selected_pages = _summary_table_page_indices(normalized)
    page_tables, table_receipt = _extract_selected_tables(content, selected_pages)
    repair_receipt = residual.add_direct_pdf_residual_statistics(
        result,
        normalized,
        page_tables,
        period_type=target["period_type"],
        source_context={
            "source_kind": source["source_kind"],
            "announcement_id": source["announcement_id"],
            "official_pdf_url": source["official_pdf_url"],
            "official_pdf_sha256": observed_sha256,
        },
    )
    return {
        **source,
        "observed_pdf_sha256": observed_sha256,
        "observed_pdf_size_bytes": len(content),
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "text_page_error_count": len(text_receipt.get("page_errors") or []),
        "table_receipt": table_receipt,
        "repair_receipt": repair_receipt,
        "status": repair_receipt["status"],
    }


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["merged_checkpoint_path"]
    checkpoint_content = checkpoint_path.read_bytes()
    observed_checkpoint_sha256 = _sha256(checkpoint_content)
    if observed_checkpoint_sha256 != target["merged_checkpoint_sha256"]:
        raise RuntimeError(f"合并检查点哈希漂移：{target['announcement_id']}")
    result = _load_json_gzip(checkpoint_path)
    before_metrics = _metric_map(result)
    before_missing = list(result.get("missing_metrics") or [])
    source_results: list[dict[str, Any]] = []
    for source in target["document_sources"]:
        try:
            source_results.append(_run_source(result, target, source))
        except Exception as error:
            source_results.append(
                {
                    **source,
                    "status": "ERROR_RESIDUAL_DIRECT_PDF_SOURCE",
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            )
    parser._restamp_result(result)
    after_metrics = _metric_map(result)
    after_missing = list(result.get("missing_metrics") or [])
    admitted = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    corrected = [
        metric_id
        for metric_id in parser.REQUIRED_METRICS
        if metric_id in before_metrics
        and metric_id in after_metrics
        and before_metrics[metric_id] != after_metrics[metric_id]
    ]
    removed = [
        metric_id
        for metric_id in before_metrics
        if metric_id not in after_metrics
    ]
    changed = [
        metric_id
        for metric_id in parser.REQUIRED_METRICS
        if metric_id in admitted or metric_id in corrected
    ]
    return {
        "announcement_id": target["announcement_id"],
        "ts_code": target["ts_code"],
        "report_period": target["report_period"],
        "period_type": target["period_type"],
        "merged_checkpoint_path": target["merged_checkpoint_path"],
        "merged_checkpoint_sha256": target["merged_checkpoint_sha256"],
        "before_missing_metrics": before_missing,
        "after_missing_metrics": after_missing,
        "newly_admitted_metric_ids": admitted,
        "corrected_metric_ids": corrected,
        "removed_metric_ids": removed,
        "changed_metric_ids": changed,
        "changed_metrics": [after_metrics[metric_id] for metric_id in changed],
        "source_results": source_results,
        "document_complete_before_residual_statistics": len(before_missing) == 0,
        "document_complete_after_residual_statistics": len(after_missing) == 0,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _pattern_counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(
        "|".join(row.get(field) or [])
        for row in rows
        if "error" not in row and row.get(field)
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _metric_counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(
        metric_id
        for row in rows
        for metric_id in row.get(field) or []
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="逐份直读官方PDF，修复年度季度表污染并补齐高杠杆残缺事实"
    )
    argument_parser.add_argument("--workers", type=int, default=3)
    argument_parser.add_argument(
        "--announcement-id",
        action="append",
        default=[],
        help="仅处理指定公告编号；可重复传入，用于可复现回归",
    )
    argument_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)),
    )
    return argument_parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    output_root = _resolve_output_root(args.output_root)
    receipt_path = output_root / "receipt.json"
    output_root.mkdir(parents=True, exist_ok=True)

    direct_content = DIRECT_RECEIPT_PATH.read_bytes()
    direct_receipt = json.loads(direct_content.decode("utf-8"))
    direct_rows = list(direct_receipt.get("results") or [])
    direct_map = {str(row["announcement_id"]): row for row in direct_rows}

    manifest_content = PATCH_MANIFEST_PATH.read_bytes()
    manifest = pd.read_parquet(PATCH_MANIFEST_PATH)
    manifest_rows = manifest.to_dict(orient="records")
    selected_ids = {str(value) for value in args.announcement_id}
    targets: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        announcement_id = str(manifest_row["announcement_id"])
        direct_row = direct_map[announcement_id]
        final_missing = json.loads(str(manifest_row["final_missing_metrics_json"]))
        eligible = bool(final_missing) or str(manifest_row["period_type"]).upper() == "FY"
        if not eligible or (selected_ids and announcement_id not in selected_ids):
            continue
        sources = [
            contract
            for source in direct_row.get("source_results") or []
            if (contract := _source_contract(source)) is not None
        ]
        targets.append(
            {
                "announcement_id": announcement_id,
                "ts_code": str(manifest_row["ts_code"]),
                "report_period": str(manifest_row["report_period"]),
                "period_type": str(manifest_row["period_type"]),
                "merged_checkpoint_path": str(manifest_row["merged_checkpoint_path"]),
                "merged_checkpoint_sha256": str(
                    manifest_row["merged_checkpoint_sha256"]
                ),
                "document_sources": sources,
            }
        )
    targets.sort(key=lambda item: item["announcement_id"])
    if selected_ids:
        found_ids = {target["announcement_id"] for target in targets}
        unavailable = sorted(selected_ids - found_ids)
        if unavailable:
            raise ValueError("指定公告不在处理集合：" + ",".join(unavailable))

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_target, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {
                    "announcement_id": target["announcement_id"],
                    "ts_code": target["ts_code"],
                    "report_period": target["report_period"],
                    "period_type": target["period_type"],
                    "merged_checkpoint_path": target["merged_checkpoint_path"],
                    "merged_checkpoint_sha256": target["merged_checkpoint_sha256"],
                    "before_missing_metrics": [],
                    "after_missing_metrics": [],
                    "newly_admitted_metric_ids": [],
                    "corrected_metric_ids": [],
                    "removed_metric_ids": [],
                    "changed_metric_ids": [],
                    "changed_metrics": [],
                    "source_results": [],
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            print(
                f"残缺直读 {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"修正={len(row.get('corrected_metric_ids') or [])}｜"
                f"剩余={len(row.get('after_missing_metrics') or [])}",
                flush=True,
            )

    results.sort(key=lambda item: item["announcement_id"])
    result_map = {str(row["announcement_id"]): row for row in results}
    combined_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        announcement_id = str(manifest_row["announcement_id"])
        result_row = result_map.get(announcement_id)
        final_missing = json.loads(str(manifest_row["final_missing_metrics_json"]))
        combined_rows.append(
            {
                "announcement_id": announcement_id,
                "period_type": str(manifest_row["period_type"]),
                "after_missing_metrics": list(
                    result_row.get("after_missing_metrics")
                    if result_row is not None and "error" not in result_row
                    else final_missing
                ),
            }
        )

    target_exception_count = sum("error" in row for row in results)
    source_exception_count = sum(
        bool(source.get("error"))
        for row in results
        for source in row.get("source_results") or []
    )
    complete_before = sum(
        bool(row.get("document_complete_before_residual_statistics"))
        for row in results
    )
    complete_after = sum(
        bool(row.get("document_complete_after_residual_statistics"))
        for row in results
    )
    overall_complete_after = sum(
        not (row.get("after_missing_metrics") or []) for row in combined_rows
    )
    residual_period_counts = Counter(
        str(row.get("period_type") or "UNKNOWN")
        for row in combined_rows
        if row.get("after_missing_metrics")
    )
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "input_direct_receipt_path": str(DIRECT_RECEIPT_PATH.relative_to(ROOT)),
        "input_direct_receipt_sha256": _sha256(direct_content),
        "input_patch_manifest_path": str(PATCH_MANIFEST_PATH.relative_to(ROOT)),
        "input_patch_manifest_sha256": _sha256(manifest_content),
        "residual_parser_version": residual.RESIDUAL_PARSER_VERSION,
        "worker_count": args.workers,
        "explicit_announcement_ids": sorted(selected_ids),
        "input_document_count": len(manifest_rows),
        "target_count": len(results),
        "complete_before_residual_statistics_count": complete_before,
        "complete_after_residual_statistics_count": complete_after,
        "newly_completed_document_count": complete_after - complete_before,
        "newly_admitted_metric_count": sum(
            len(row.get("newly_admitted_metric_ids") or []) for row in results
        ),
        "corrected_metric_count": sum(
            len(row.get("corrected_metric_ids") or []) for row in results
        ),
        "removed_unreplaced_metric_count": sum(
            len(row.get("removed_metric_ids") or []) for row in results
        ),
        "overall_complete_document_count_after_residual_statistics": (
            overall_complete_after
        ),
        "overall_still_incomplete_count_after_residual_statistics": (
            len(combined_rows) - overall_complete_after
        ),
        "target_exception_count": target_exception_count,
        "source_exception_count": source_exception_count,
        "before_missing_pattern_counts": _pattern_counts(
            results,
            "before_missing_metrics",
        ),
        "after_missing_pattern_counts": _pattern_counts(
            results,
            "after_missing_metrics",
        ),
        "newly_admitted_metric_counts": _metric_counts(
            results,
            "newly_admitted_metric_ids",
        ),
        "corrected_metric_counts": _metric_counts(results, "corrected_metric_ids"),
        "overall_residual_missing_pattern_counts": _pattern_counts(
            combined_rows,
            "after_missing_metrics",
        ),
        "overall_residual_missing_metric_counts": _metric_counts(
            combined_rows,
            "after_missing_metrics",
        ),
        "overall_residual_period_type_counts": dict(
            sorted(
                residual_period_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(receipt_path, receipt)
    print(
        "残缺直读完成："
        f"目标={len(results)}；"
        f"新增完整文档={receipt['newly_completed_document_count']}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"修正指标={receipt['corrected_metric_count']}；"
        f"全体完整={overall_complete_after}/{len(combined_rows)}；"
        f"目标异常={target_exception_count}；"
        f"来源异常={source_exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if target_exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
