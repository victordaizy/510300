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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_ocr_v1 as direct_ocr  # noqa: E402
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402
from scripts import census_csi300_pit_official_facts_v1_7_targeted_direct_pdf_ocr as legacy  # noqa: E402


DIRECT_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_statistics/receipt.json"
COMPANION_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_companions/receipt.json"
DEFAULT_INPUT_QUEUE_PATH = ROOT / (
    "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "official_fact_document_queue_v1_9_receivable_direct_pdf.parquet"
)
DEFAULT_OUTPUT_ROOT = ROOT / "tmp/pit_current_targeted_direct_pdf_ocr"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_CURRENT_TARGETED_DIRECT_PDF_OCR_CENSUS_V1"
VERIFIED_CURRENT_COMPANION_SOURCES: dict[str, dict[str, str]] = {
    "1207684806": {
        "ts_code": "000333.SZ",
        "report_period": "2020-03-31",
        "period_type": "Q1",
        "publication_date": "2020-04-30",
        "companion_announcement_id": "1207684809",
        "companion_announcement_title": "2020年第一季度报告全文",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2020-04-30/1207684809.PDF",
        "path": "tmp/pit_v1_7_direct_pdf_companions/1207684806__1207684809.pdf",
        "expected_pdf_sha256": "2a997a4d3243b66d30a7de6fda1f60fa264f679b5f5d1dcddad84ede38e9263e",
    },
    "1207688732": {
        "ts_code": "600115.SH",
        "report_period": "2020-03-31",
        "period_type": "Q1",
        "publication_date": "2020-04-30",
        "companion_announcement_id": "1207688733",
        "companion_announcement_title": "2020年第一季度报告",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2020-04-30/1207688733.PDF",
        "path": "tmp/pit_v1_7_direct_pdf_companions/1207688732__1207688733.pdf",
        "expected_pdf_sha256": "946e431cc9a2d19adf27b9e20e78bbe6e182ba1224f55d140fd7713e568ec3fb",
    },
    "1207690237": {
        "ts_code": "600426.SH",
        "report_period": "2020-03-31",
        "period_type": "Q1",
        "publication_date": "2020-04-30",
        "companion_announcement_id": "1207690238",
        "companion_announcement_title": "2020年第一季度报告",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2020-04-30/1207690238.PDF",
        "path": "tmp/pit_v1_7_direct_pdf_companions/1207690237__1207690238.pdf",
        "expected_pdf_sha256": "cc5753bdeae807792c335fc279b57f8b9c65082ba1f13d1b2c0a9f2e2d4f86fe",
    },
    "1225064850": {
        "ts_code": "600115.SH",
        "report_period": "2025-12-31",
        "period_type": "FY",
        "publication_date": "2026-03-31",
        "companion_announcement_id": "1225064857",
        "companion_announcement_title": "中国东方航空股份有限公司2025年度报告",
        "official_pdf_url": "https://static.cninfo.com.cn/finalpage/2026-03-31/1225064857.PDF",
        "path": "tmp/pit_v1_7_direct_pdf_companions/1225064850__1225064857.pdf",
        "expected_pdf_sha256": "dd6293baf3cdd9c53c95b154eed9fd58c042b4d05bf398c35cbe54490b5fb7d8",
    },
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _missing_metrics(row: dict[str, Any]) -> list[str]:
    raw = row.get("missing_metrics_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    values = json.loads(str(raw))
    if not isinstance(values, list):
        raise ValueError(
            f"缺失指标字段不是列表：{row.get('announcement_id')}"
        )
    return [str(value) for value in values]


def _companion_source_contract(row: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = str(row.get("selected_companion_announcement_id") or "")
    if not selected_id:
        return None
    selected = next(
        (
            candidate
            for candidate in row.get("candidate_results") or []
            if str(candidate.get("companion_announcement_id") or "") == selected_id
        ),
        None,
    )
    if selected is None:
        return None
    if not all(
        bool(selected.get(field))
        for field in (
            "same_company",
            "same_period_type_from_title",
            "same_publication_date",
        )
    ):
        return None
    path = str(selected.get("path") or "")
    expected_sha256 = str(
        selected.get("observed_pdf_sha256")
        or row.get("selected_companion_pdf_sha256")
        or ""
    )
    official_pdf_url = str(
        selected.get("companion_official_pdf_url")
        or row.get("selected_companion_official_pdf_url")
        or ""
    )
    if not path or not expected_sha256 or not official_pdf_url:
        return None
    return {
        "source_kind": "SAME_DAY_OFFICIAL_PERIODIC_REPORT_COMPANION",
        "announcement_id": selected_id,
        "official_pdf_url": official_pdf_url,
        "path": path,
        "expected_pdf_sha256": expected_sha256,
    }


def _verified_current_companion_source_contract(
    row: dict[str, Any],
) -> dict[str, Any] | None:
    source_announcement_id = str(row["announcement_id"])
    source = VERIFIED_CURRENT_COMPANION_SOURCES.get(source_announcement_id)
    if source is None:
        return None
    expected_identity = {
        "ts_code": str(row["ts_code"]),
        "report_period": str(row["report_period"]),
        "period_type": str(row["period_type"]),
    }
    observed_identity = {
        field: str(source[field]) for field in expected_identity
    }
    if observed_identity != expected_identity:
        raise ValueError(
            "已核验同日官方全文身份与当前队列不一致："
            f"{source_announcement_id}|expected={expected_identity}|"
            f"observed={observed_identity}"
        )
    path = _resolve_path(source["path"])
    content = path.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"已核验同日官方全文不是PDF：{path}")
    observed_sha256 = _sha256(content)
    if observed_sha256 != source["expected_pdf_sha256"]:
        raise ValueError(
            "已核验同日官方全文哈希漂移："
            f"{source_announcement_id}|expected={source['expected_pdf_sha256']}|"
            f"observed={observed_sha256}"
        )
    return {
        "source_kind": "SAME_DAY_OFFICIAL_PERIODIC_REPORT_COMPANION",
        "announcement_id": source["companion_announcement_id"],
        "official_pdf_url": source["official_pdf_url"],
        "path": source["path"],
        "expected_pdf_sha256": source["expected_pdf_sha256"],
        "source_contract": {
            "source_announcement_id": source_announcement_id,
            "same_company": True,
            "same_report_period": True,
            "same_period_type": True,
            "same_publication_date": True,
            "publication_date": source["publication_date"],
            "companion_announcement_title": source[
                "companion_announcement_title"
            ],
        },
    }


def _load_current_checkpoint(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["checkpoint_path"]
    content = checkpoint_path.read_bytes()
    if _sha256(content) != target["checkpoint_sha256"]:
        raise RuntimeError(f"检查点哈希漂移：{target['announcement_id']}")
    with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
        result = json.load(handle)
    observed_missing = list(result.get("missing_metrics") or [])
    if observed_missing != target["expected_current_missing_metrics"]:
        raise RuntimeError(
            "当前检查点缺失状态不一致："
            f"{target['announcement_id']}|"
            f"expected={target['expected_current_missing_metrics']}|"
            f"observed={observed_missing}"
        )
    return result


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    result = _load_current_checkpoint(target)
    before_missing = list(result.get("missing_metrics") or [])
    source_results: list[dict[str, Any]] = []
    for source in target["document_sources"]:
        if target.get("direct_text_only"):
            if not result.get("missing_metrics"):
                break
        elif not direct_ocr.TARGET_METRICS.intersection(
            result.get("missing_metrics") or []
        ):
            break
        try:
            if target.get("direct_text_only"):
                source_results.append(
                    _run_source_direct_text_only(result, target, source)
                )
            else:
                source_results.append(legacy._run_source(result, target, source))
        except Exception as error:
            source_results.append(
                {
                    **source,
                    "status": "ERROR_CURRENT_TARGETED_OCR_SOURCE",
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
        "announcement_id": target["announcement_id"],
        "ts_code": target["ts_code"],
        "report_period": target["report_period"],
        "period_type": target["period_type"],
        "checkpoint_path": target["checkpoint_path"],
        "checkpoint_sha256": target["checkpoint_sha256"],
        "before_missing_metrics": before_missing,
        "after_missing_metrics": after_missing,
        "newly_admitted_metric_ids": admitted,
        "new_metrics": [metric_map[metric_id] for metric_id in admitted],
        "source_results": source_results,
        "document_complete_before_targeted_ocr": not before_missing,
        "document_complete_after_targeted_ocr": not after_missing,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _run_source_direct_text_only(
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
            "repair_phase": "CURRENT_RESIDUAL_DIRECT_TEXT_ONLY",
        },
    )
    return {
        **source,
        "observed_pdf_sha256": observed_sha256,
        "observed_pdf_size_bytes": len(content),
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
        "status": direct_text_receipt["status"],
        "direct_text_repair_receipt": direct_text_receipt,
        "selection_receipt": None,
        "ocr_receipt": None,
        "repair_receipt": direct_text_receipt,
    }


def _error_row(target: dict[str, Any], error: Exception) -> dict[str, Any]:
    missing = list(target["expected_current_missing_metrics"])
    return {
        "announcement_id": target["announcement_id"],
        "ts_code": target["ts_code"],
        "report_period": target["report_period"],
        "period_type": target["period_type"],
        "checkpoint_path": target["checkpoint_path"],
        "checkpoint_sha256": target["checkpoint_sha256"],
        "before_missing_metrics": missing,
        "after_missing_metrics": missing,
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


def _build_targets(
    queue_rows: list[dict[str, Any]],
    direct_map: dict[str, dict[str, Any]],
    companion_map: dict[str, dict[str, Any]],
    selected_ids: set[str],
    *,
    direct_text_only: bool,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in queue_rows:
        announcement_id = str(row["announcement_id"])
        missing = _missing_metrics(row)
        if not missing:
            continue
        if not direct_text_only and not direct_ocr.TARGET_METRICS.intersection(missing):
            continue
        if selected_ids and announcement_id not in selected_ids:
            continue
        direct_row = direct_map.get(announcement_id)
        if direct_row is None:
            raise ValueError(f"直读来源收据缺少公告：{announcement_id}")
        sources = [
            contract
            for source in direct_row.get("source_results") or []
            if (contract := legacy._source_contract(source)) is not None
        ]
        companion_row = companion_map.get(announcement_id)
        if companion_row is not None:
            companion_contract = _companion_source_contract(companion_row)
            if companion_contract is not None and all(
                source["announcement_id"] != companion_contract["announcement_id"]
                for source in sources
            ):
                sources.append(companion_contract)
        verified_current_companion = _verified_current_companion_source_contract(row)
        if verified_current_companion is not None and all(
            source["announcement_id"]
            != verified_current_companion["announcement_id"]
            for source in sources
        ):
            sources.append(verified_current_companion)
        if not sources:
            raise ValueError(f"公告没有可复用的官方PDF来源：{announcement_id}")
        targets.append(
            {
                "announcement_id": announcement_id,
                "ts_code": str(row["ts_code"]),
                "report_period": str(row["report_period"]),
                "period_type": str(row["period_type"]),
                "checkpoint_path": str(row["checkpoint_path"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "expected_current_missing_metrics": missing,
                "document_sources": sources,
                "direct_text_only": direct_text_only,
            }
        )
    targets.sort(key=lambda item: item["announcement_id"])
    if selected_ids:
        found_ids = {target["announcement_id"] for target in targets}
        unavailable = sorted(selected_ids - found_ids)
        if unavailable:
            raise ValueError(
                "指定公告不在当前定向OCR集合：" + ",".join(unavailable)
            )
    return targets


def _combined_rows(
    queue_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_map = {str(row["announcement_id"]): row for row in results}
    combined: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        announcement_id = str(queue_row["announcement_id"])
        missing = _missing_metrics(queue_row)
        result_row = result_map.get(announcement_id)
        if result_row is not None and "error" not in result_row:
            missing = list(result_row.get("after_missing_metrics") or [])
        combined.append(
            {
                "announcement_id": announcement_id,
                "period_type": str(queue_row["period_type"]),
                "after_missing_metrics": missing,
            }
        )
    return combined


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="仅对最新队列的残缺官方PDF执行双尺度定向OCR"
    )
    argument_parser.add_argument("--workers", type=int, default=3)
    argument_parser.add_argument(
        "--announcement-id",
        action="append",
        default=[],
        help="仅处理指定公告编号；可重复传入",
    )
    argument_parser.add_argument(
        "--input-queue",
        default=str(DEFAULT_INPUT_QUEUE_PATH.relative_to(ROOT)),
    )
    argument_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)),
    )
    argument_parser.add_argument(
        "--direct-text-only",
        action="store_true",
        help="仅直读官方PDF文本层，不执行页面渲染或OCR",
    )
    return argument_parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    input_queue_path = _resolve_path(args.input_queue)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    queue_content = input_queue_path.read_bytes()
    queue = pd.read_parquet(input_queue_path)
    queue_rows = queue.to_dict(orient="records")
    direct_content = DIRECT_RECEIPT_PATH.read_bytes()
    direct_receipt = json.loads(direct_content.decode("utf-8"))
    direct_map = {
        str(row["announcement_id"]): row
        for row in direct_receipt.get("results") or []
    }
    companion_content = COMPANION_RECEIPT_PATH.read_bytes()
    companion_receipt = json.loads(companion_content.decode("utf-8"))
    companion_map = {
        str(row["source_announcement_id"]): row
        for row in companion_receipt.get("results") or []
    }
    selected_ids = {str(value) for value in args.announcement_id}
    targets = _build_targets(
        queue_rows,
        direct_map,
        companion_map,
        selected_ids,
        direct_text_only=bool(args.direct_text_only),
    )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_target, target): target for target in targets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = _error_row(target, error)
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
                f"当前定向OCR {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜页={selected_page_count}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"剩余={len(row.get('after_missing_metrics') or [])}",
                flush=True,
            )

    results.sort(key=lambda item: item["announcement_id"])
    combined_rows = _combined_rows(queue_rows, results)
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
        "input_current_queue_path": str(input_queue_path.relative_to(ROOT)),
        "input_current_queue_sha256": _sha256(queue_content),
        "input_direct_receipt_path": str(DIRECT_RECEIPT_PATH.relative_to(ROOT)),
        "input_direct_receipt_sha256": _sha256(direct_content),
        "input_companion_receipt_path": str(
            COMPANION_RECEIPT_PATH.relative_to(ROOT)
        ),
        "input_companion_receipt_sha256": _sha256(companion_content),
        "direct_ocr_version": direct_ocr.DIRECT_OCR_VERSION,
        "direct_text_only": bool(args.direct_text_only),
        "worker_count": args.workers,
        "explicit_announcement_ids": sorted(selected_ids),
        "input_document_count": len(queue_rows),
        "input_incomplete_document_count": sum(
            bool(_missing_metrics(row)) for row in queue_rows
        ),
        "target_count": len(results),
        "complete_before_targeted_ocr_count": complete_before,
        "complete_after_targeted_ocr_count": complete_after,
        "newly_completed_document_count": complete_after - complete_before,
        "newly_admitted_metric_count": sum(
            len(row.get("newly_admitted_metric_ids") or []) for row in results
        ),
        "overall_complete_document_count_after_targeted_ocr": (
            overall_complete_after
        ),
        "overall_still_incomplete_count_after_targeted_ocr": (
            len(combined_rows) - overall_complete_after
        ),
        "target_exception_count": target_exception_count,
        "source_exception_count": source_exception_count,
        "ocr_source_run_count": sum(
            source.get("ocr_receipt") is not None
            for row in results
            for source in row.get("source_results") or []
        ),
        "selected_page_count": sum(
            int(
                (source.get("selection_receipt") or {}).get(
                    "selected_page_count"
                )
                or 0
            )
            for row in results
            for source in row.get("source_results") or []
        ),
        "selection_route_counts": legacy._selection_route_counts(results),
        "target_before_missing_pattern_counts": legacy._pattern_counts(
            results,
            "before_missing_metrics",
        ),
        "target_after_missing_pattern_counts": legacy._pattern_counts(
            results,
            "after_missing_metrics",
        ),
        "newly_admitted_metric_counts": legacy._metric_counts(
            results,
            "newly_admitted_metric_ids",
        ),
        "overall_residual_missing_pattern_counts": legacy._pattern_counts(
            combined_rows,
            "after_missing_metrics",
        ),
        "overall_residual_missing_metric_counts": legacy._metric_counts(
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
    legacy._atomic_write_json(output_root / "receipt.json", receipt)
    print(
        "当前定向OCR完成："
        f"目标={len(results)}；"
        f"新增完整文档={receipt['newly_completed_document_count']}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"全体完整={overall_complete_after}/{len(combined_rows)}；"
        f"目标异常={target_exception_count}；"
        f"来源异常={source_exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if target_exception_count == 0 and source_exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
