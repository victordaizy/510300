from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_ocr_v1 as direct_ocr  # noqa: E402
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402


DIRECT_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_statistics/receipt.json"
DEFAULT_OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_targeted_direct_pdf_ocr"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_TARGETED_DIRECT_PDF_OCR_CENSUS_V1"


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _target_from_direct_row(row: dict[str, Any]) -> dict[str, Any]:
    sources = [
        contract
        for source in row.get("source_results") or []
        if (contract := _source_contract(source)) is not None
    ]
    return {
        "announcement_id": str(row["announcement_id"]),
        "ts_code": str(row["ts_code"]),
        "report_period": str(row["report_period"]),
        "period_type": str(row["period_type"]),
        "checkpoint_path": str(row["checkpoint_path"]),
        "checkpoint_sha256": str(row["checkpoint_sha256"]),
        "all_gap_new_metrics": list(row.get("all_gap_new_metrics") or []),
        "companion_new_metrics": list(row.get("companion_new_metrics") or []),
        "direct_new_metrics": list(row.get("new_metrics") or []),
        "expected_direct_after_missing_metrics": list(
            row.get("after_missing_metrics") or []
        ),
        "document_sources": sources,
    }


def _baseline_result(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["checkpoint_path"]
    checkpoint_content = checkpoint_path.read_bytes()
    observed_checkpoint_sha256 = _sha256(checkpoint_content)
    if observed_checkpoint_sha256 != target["checkpoint_sha256"]:
        raise RuntimeError(f"检查点哈希漂移：{target['announcement_id']}")
    result = _load_json_gzip(checkpoint_path)
    for metric in (
        list(target["all_gap_new_metrics"])
        + list(target["companion_new_metrics"])
        + list(target["direct_new_metrics"])
    ):
        if metric["metric_id"] not in result.get("missing_metrics", []):
            continue
        parser._v1_4._put_metric(
            result,
            parser._base.MetricEvidence(**metric),
        )
    parser._restamp_result(result)
    observed_missing = list(result.get("missing_metrics") or [])
    if observed_missing != target["expected_direct_after_missing_metrics"]:
        raise RuntimeError(
            "直接统计基线复算不一致："
            f"{target['announcement_id']}|"
            f"expected={target['expected_direct_after_missing_metrics']}|"
            f"observed={observed_missing}"
        )
    return result


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
    direct_text_receipt = direct_ocr.direct.add_direct_pdf_document_statistics(
        result,
        page_texts,
        period_type=target["period_type"],
        source_context={
            "source_kind": source["source_kind"],
            "announcement_id": source["announcement_id"],
            "official_pdf_url": source["official_pdf_url"],
            "official_pdf_sha256": observed_sha256,
            "repair_phase": "DIRECT_TEXT_BEFORE_TARGETED_OCR",
        },
    )
    pages, selection_receipt = direct_ocr.select_targeted_ocr_pages(
        result,
        page_texts,
        period_type=target["period_type"],
    )
    source_announcement_id = str(source["announcement_id"])
    manual_page_numbers = (
        direct_ocr.MANUAL_FINANCIAL_STATEMENT_PAGE_NUMBERS.get(
            source_announcement_id,
            (),
        )
    )
    if manual_page_numbers:
        invalid_page_numbers = [
            page_number
            for page_number in manual_page_numbers
            if page_number < 1 or page_number > len(page_texts)
        ]
        if invalid_page_numbers:
            raise ValueError(
                "人工定位财务报表页超出PDF范围："
                f"{target['announcement_id']}|"
                f"pages={invalid_page_numbers}|pdf_pages={len(page_texts)}"
            )
        manual_pages = tuple(page_number - 1 for page_number in manual_page_numbers)
        if source_announcement_id in (
            direct_ocr.MANUAL_EXCLUSIVE_OCR_PAGE_ANNOUNCEMENT_IDS
        ):
            pages = tuple(sorted(set(manual_pages)))
            manual_route = "MANUAL_VISUAL_EXCLUSIVE_SINGLE_PAGE_LOCATOR"
        else:
            pages = tuple(sorted(set(pages) | set(manual_pages)))
            manual_route = "MANUAL_VISUAL_FINANCIAL_STATEMENT_PAGE_LOCATOR"
        selection_receipt["selection_routes"] = list(
            selection_receipt.get("selection_routes") or []
        ) + [manual_route]
        selection_receipt["manual_page_numbers"] = list(manual_page_numbers)
        selection_receipt["selected_page_numbers"] = [
            page + 1 for page in pages
        ]
        selection_receipt["selected_page_count"] = len(pages)
    common = {
        **source,
        "observed_pdf_sha256": observed_sha256,
        "observed_pdf_size_bytes": len(content),
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
        "direct_text_repair_receipt": direct_text_receipt,
        "selection_receipt": selection_receipt,
    }
    if not direct_ocr.TARGET_METRICS.intersection(
        result.get("missing_metrics") or []
    ):
        return {
            **common,
            "status": "PASS_DIRECT_TEXT_REPAIR_NO_TARGETED_OCR_REQUIRED",
            "ocr_receipt": None,
            "repair_receipt": {
                "status": "PASS_DIRECT_TEXT_REPAIR_NO_TARGETED_OCR_REQUIRED",
                "direct_text": direct_text_receipt,
                "admitted_metric_ids": list(
                    direct_text_receipt.get("admitted_metric_ids") or []
                ),
                "market_price_read": False,
                "future_return_read": False,
                "return_evaluation": "NOT_ALLOWED",
            },
        }
    if not pages:
        return {
            **common,
            "status": "NO_VIEW_NO_TARGETED_OCR_PAGE_SELECTION",
            "ocr_receipt": None,
            "repair_receipt": {
                "status": "NO_MATCH_NO_TARGETED_OCR_PAGE_SELECTION",
                "admitted_metric_ids": [],
                "market_price_read": False,
                "future_return_read": False,
                "return_evaluation": "NOT_ALLOWED",
            },
        }
    outputs, ocr_receipt = parser._v1_6._v1_3._run_windows_ocr(
        content,
        pages,
        render_scales=direct_ocr.DIRECT_OCR_RENDER_SCALES,
    )
    note_pages, note_selection_receipt = (
        direct_ocr.select_residual_note_ocr_pages(
            result,
            page_texts,
            outputs,
        )
    )
    additional_note_pages = tuple(
        page for page in note_pages if page not in set(pages)
    )
    if additional_note_pages:
        note_outputs, note_ocr_receipt = (
            parser._v1_6._v1_3._run_windows_ocr(
                content,
                additional_note_pages,
                render_scales=direct_ocr.DIRECT_OCR_RENDER_SCALES,
            )
        )
        for scale, records in note_outputs.items():
            outputs[scale].extend(records)
            outputs[scale].sort(key=lambda record: int(record["page_index"]))
        all_pages = tuple(sorted(set(pages) | set(additional_note_pages)))
        ocr_receipt = {
            "status": "PASS_WINDOWS_OCR_RENDERED_TWO_PASS",
            "engine": ocr_receipt.get("engine"),
            "language_tag": ocr_receipt.get("language_tag"),
            "max_image_dimension": ocr_receipt.get("max_image_dimension"),
            "render_scales": list(direct_ocr.DIRECT_OCR_RENDER_SCALES),
            "page_numbers": [page + 1 for page in all_pages],
            "initial_pass": ocr_receipt,
            "note_pass": note_ocr_receipt,
            "temporary_images_retained": False,
        }
        selection_receipt["selection_routes"] = list(
            selection_receipt.get("selection_routes") or []
        ) + ["OCR_STATEMENT_NOTE_REFERENCE_TO_DIRECT_TEXT_NOTE_PAGE"]
        selection_receipt["selected_page_numbers"] = [
            page + 1 for page in all_pages
        ]
        selection_receipt["selected_page_count"] = len(all_pages)
    selection_receipt["note_selection_receipt"] = note_selection_receipt
    repair_receipt = direct_ocr.add_targeted_direct_pdf_ocr_statistics(
        result,
        outputs,
        source_context={
            "source_kind": source["source_kind"],
            "announcement_id": source["announcement_id"],
            "official_pdf_url": source["official_pdf_url"],
            "official_pdf_sha256": observed_sha256,
            "note_selection_receipt": note_selection_receipt,
        },
    )
    return {
        **common,
        "status": str(repair_receipt["status"]),
        "ocr_receipt": ocr_receipt,
        "repair_receipt": repair_receipt,
    }


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    result = _baseline_result(target)
    before_missing = list(result.get("missing_metrics") or [])
    source_results: list[dict[str, Any]] = []
    for source in target["document_sources"]:
        if not direct_ocr.TARGET_METRICS.intersection(
            result.get("missing_metrics") or []
        ):
            break
        try:
            source_results.append(_run_source(result, target, source))
        except Exception as error:  # 单附件失败不阻断同一事件的其他官方附件
            source_results.append(
                {
                    **source,
                    "status": "ERROR_TARGETED_OCR_SOURCE",
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            )
    parser._restamp_result(result)
    after_missing = list(result.get("missing_metrics") or [])
    admitted = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    metric_map = {
        str(metric["metric_id"]): metric for metric in result.get("metrics") or []
    }
    return {
        **{
            key: value
            for key, value in target.items()
            if key
            not in {
                "document_sources",
                "all_gap_new_metrics",
                "companion_new_metrics",
                "direct_new_metrics",
            }
        },
        "before_missing_metrics": before_missing,
        "source_results": source_results,
        "newly_admitted_metric_ids": admitted,
        "new_metrics": [metric_map[metric_id] for metric_id in admitted],
        "after_missing_metrics": after_missing,
        "document_complete_before_targeted_ocr": len(before_missing) == 0,
        "document_complete_after_targeted_ocr": len(after_missing) == 0,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _pattern_counts(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts = Counter(
        "|".join(row.get(field) or [])
        for row in rows
        if "error" not in row and row.get(field)
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _metric_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(
        metric_id
        for row in rows
        for metric_id in row.get(field) or []
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _selection_route_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in results:
        for source in row.get("source_results") or []:
            for route in (source.get("selection_receipt") or {}).get(
                "selection_routes"
            ) or []:
                counts[str(route)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="逐份渲染官方PDF并以双尺度Windows OCR补齐关键财务事实"
    )
    argument_parser.add_argument("--workers", type=int, default=3)
    argument_parser.add_argument(
        "--announcement-id",
        action="append",
        default=[],
        help="仅处理指定公告编号；可重复传入，仅用于可复现回归",
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

    direct_receipt_content = DIRECT_RECEIPT_PATH.read_bytes()
    direct_receipt = json.loads(direct_receipt_content.decode("utf-8"))
    direct_rows = list(direct_receipt.get("results") or [])
    selected_ids = {str(value) for value in args.announcement_id}
    eligible_rows = [
        row
        for row in direct_rows
        if direct_ocr.TARGET_METRICS.intersection(
            row.get("after_missing_metrics") or []
        )
        and (not selected_ids or str(row["announcement_id"]) in selected_ids)
    ]
    targets = sorted(
        (_target_from_direct_row(row) for row in eligible_rows),
        key=lambda item: item["announcement_id"],
    )
    if selected_ids:
        found_ids = {target["announcement_id"] for target in targets}
        unavailable = sorted(selected_ids - found_ids)
        if unavailable:
            raise ValueError(
                "指定公告不在定向OCR待处理集合：" + ",".join(unavailable)
            )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_target, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:  # 每份异常必须进入总收据
                row = {
                    "announcement_id": target["announcement_id"],
                    "ts_code": target["ts_code"],
                    "report_period": target["report_period"],
                    "period_type": target["period_type"],
                    "expected_direct_after_missing_metrics": target[
                        "expected_direct_after_missing_metrics"
                    ],
                    "before_missing_metrics": target[
                        "expected_direct_after_missing_metrics"
                    ],
                    "after_missing_metrics": target[
                        "expected_direct_after_missing_metrics"
                    ],
                    "newly_admitted_metric_ids": [],
                    "new_metrics": [],
                    "source_results": [],
                    "document_complete_before_targeted_ocr": False,
                    "document_complete_after_targeted_ocr": False,
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            selected_page_count = sum(
                int(
                    (source.get("selection_receipt") or {}).get(
                        "selected_page_count"
                    )
                    or 0
                )
                for source in row.get("source_results") or []
            )
            print(
                f"定向OCR {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜页={selected_page_count}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"完整={row.get('document_complete_after_targeted_ocr')}",
                flush=True,
            )

    results.sort(key=lambda item: item["announcement_id"])
    result_map = {str(row["announcement_id"]): row for row in results}
    combined_rows: list[dict[str, Any]] = []
    for direct_row in direct_rows:
        announcement_id = str(direct_row["announcement_id"])
        ocr_row = result_map.get(announcement_id)
        combined_rows.append(
            {
                "announcement_id": announcement_id,
                "period_type": direct_row.get("period_type"),
                "after_missing_metrics": list(
                    (
                        ocr_row.get("after_missing_metrics")
                        if ocr_row is not None
                        else direct_row.get("after_missing_metrics")
                    )
                    or []
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
        bool(row.get("document_complete_before_targeted_ocr")) for row in results
    )
    complete_after = sum(
        bool(row.get("document_complete_after_targeted_ocr")) for row in results
    )
    direct_complete_count = sum(
        not (row.get("after_missing_metrics") or []) for row in direct_rows
    )
    overall_complete_after = sum(
        not (row.get("after_missing_metrics") or []) for row in combined_rows
    )
    ocr_source_run_count = sum(
        source.get("ocr_receipt") is not None
        for row in results
        for source in row.get("source_results") or []
    )
    selected_page_count = sum(
        int(
            (source.get("selection_receipt") or {}).get("selected_page_count")
            or 0
        )
        for row in results
        for source in row.get("source_results") or []
    )
    residual_period_type_counts = Counter(
        str(row.get("period_type") or "UNKNOWN")
        for row in combined_rows
        if row.get("after_missing_metrics")
    )
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "input_direct_receipt_path": str(DIRECT_RECEIPT_PATH.relative_to(ROOT)),
        "input_direct_receipt_sha256": _sha256(direct_receipt_content),
        "base_parser_version": parser.PARSER_VERSION,
        "direct_ocr_version": direct_ocr.DIRECT_OCR_VERSION,
        "worker_count": args.workers,
        "explicit_announcement_ids": sorted(selected_ids),
        "input_document_count": len(direct_rows),
        "target_count": len(results),
        "complete_before_targeted_ocr_count": complete_before,
        "complete_after_targeted_ocr_count": complete_after,
        "newly_completed_document_count": complete_after - complete_before,
        "newly_admitted_metric_count": sum(
            len(row.get("newly_admitted_metric_ids") or []) for row in results
        ),
        "direct_complete_document_count": direct_complete_count,
        "overall_complete_document_count_after_targeted_ocr": (
            overall_complete_after
        ),
        "overall_still_incomplete_count_after_targeted_ocr": (
            len(combined_rows) - overall_complete_after
        ),
        "target_exception_count": target_exception_count,
        "source_exception_count": source_exception_count,
        "ocr_source_run_count": ocr_source_run_count,
        "selected_page_count": selected_page_count,
        "selection_route_counts": _selection_route_counts(results),
        "target_before_missing_pattern_counts": _pattern_counts(
            results,
            "before_missing_metrics",
        ),
        "target_after_missing_pattern_counts": _pattern_counts(
            results,
            "after_missing_metrics",
        ),
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
                residual_period_type_counts.items(),
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
        "定向OCR完成："
        f"目标={len(results)}；"
        f"新增完整文档={receipt['newly_completed_document_count']}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"全体完整={overall_complete_after}/{len(combined_rows)}；"
        f"目标异常={target_exception_count}；"
        f"来源异常={source_exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if target_exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
