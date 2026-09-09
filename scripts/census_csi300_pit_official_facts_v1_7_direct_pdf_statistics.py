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


from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_v1 as direct  # noqa: E402
from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402


ALL_GAP_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_all_gap_census/receipt.json"
COMPANION_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_companions/receipt.json"
OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_direct_pdf_statistics"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")


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


def _selected_companion_source(
    companion_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if companion_row is None:
        return None
    selected_id = str(
        companion_row.get("selected_companion_announcement_id") or ""
    )
    if not selected_id:
        return None
    selected = next(
        (
            candidate
            for candidate in companion_row.get("candidate_results") or []
            if str(candidate.get("companion_announcement_id") or "") == selected_id
        ),
        None,
    )
    if selected is None or selected.get("error"):
        return None
    if selected.get("overlap_mismatches"):
        return None
    path = selected.get("path")
    sha256 = selected.get("observed_pdf_sha256")
    if not path or not sha256:
        return None
    return {
        "source_kind": "SAME_DAY_OFFICIAL_FULL_TEXT_COMPANION",
        "announcement_id": selected_id,
        "official_pdf_url": selected.get("companion_official_pdf_url"),
        "path": str(path),
        "expected_pdf_sha256": str(sha256),
    }


def _targets() -> list[dict[str, Any]]:
    all_gap = _load_json(ALL_GAP_RECEIPT_PATH)
    companion = (
        _load_json(COMPANION_RECEIPT_PATH)
        if COMPANION_RECEIPT_PATH.exists()
        else {"results": []}
    )
    companion_map = {
        str(row["source_announcement_id"]): row
        for row in companion.get("results") or []
    }
    targets: list[dict[str, Any]] = []
    for row in all_gap["results"]:
        announcement_id = str(row["announcement_id"])
        companion_row = companion_map.get(announcement_id)
        sources = [
            {
                "source_kind": "CHECKPOINT_OFFICIAL_ORIGINAL_PDF",
                "announcement_id": announcement_id,
                "official_pdf_url": row["official_pdf_url"],
                "path": row["path"],
                "expected_pdf_sha256": row["observed_pdf_sha256"],
            }
        ]
        selected_companion = _selected_companion_source(companion_row)
        if (
            selected_companion is not None
            and selected_companion["expected_pdf_sha256"]
            != sources[0]["expected_pdf_sha256"]
        ):
            sources.append(selected_companion)
        targets.append(
            {
                "announcement_id": announcement_id,
                "ts_code": row["ts_code"],
                "report_period": row["report_period"],
                "period_type": row["period_type"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "all_gap_new_metrics": list(row.get("new_metrics") or []),
                "companion_new_metrics": list(
                    (companion_row or {}).get("selected_new_metrics") or []
                ),
                "document_sources": sources,
            }
        )
    return sorted(targets, key=lambda item: item["announcement_id"])


def _baseline_result(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["checkpoint_path"]
    checkpoint_content = checkpoint_path.read_bytes()
    if _sha256(checkpoint_content) != target["checkpoint_sha256"]:
        raise RuntimeError(f"检查点哈希漂移：{target['announcement_id']}")
    result = _load_json_gzip(checkpoint_path)
    for metric in (
        list(target["all_gap_new_metrics"])
        + list(target["companion_new_metrics"])
    ):
        if metric["metric_id"] not in result.get("missing_metrics", []):
            continue
        parser._v1_4._put_metric(
            result,
            parser._base.MetricEvidence(**metric),
        )
    parser._restamp_result(result)
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
            f"官方PDF哈希漂移：{target['announcement_id']}|{source['announcement_id']}"
        )
    if not content.startswith(b"%PDF-"):
        raise RuntimeError(
            f"官方附件没有PDF文件头：{target['announcement_id']}|"
            f"{source['announcement_id']}"
        )
    page_texts, text_receipt = parser._base.extract_pdf_page_texts(content)
    repair_receipt = direct.add_direct_pdf_document_statistics(
        result,
        page_texts,
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
        "page_error_count": len(text_receipt.get("page_errors") or []),
        "repair_receipt": repair_receipt,
    }


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    result = _baseline_result(target)
    before_missing = list(result.get("missing_metrics") or [])
    source_results: list[dict[str, Any]] = []
    for source in target["document_sources"]:
        if not result.get("missing_metrics"):
            break
        try:
            source_results.append(_run_source(result, target, source))
        except Exception as error:  # 单个附件失败不阻断同一事件的其他官方附件
            source_results.append(
                {
                    **source,
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
        **{key: value for key, value in target.items() if key != "document_sources"},
        "before_missing_metrics": before_missing,
        "source_results": source_results,
        "newly_admitted_metric_ids": admitted,
        "new_metrics": [metric_map[metric_id] for metric_id in admitted],
        "after_missing_metrics": after_missing,
        "document_complete_before_direct_statistics": len(before_missing) == 0,
        "document_complete_after_direct_statistics": len(after_missing) == 0,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _pattern_counts(
    results: list[dict[str, Any]],
    field: str,
) -> dict[str, int]:
    counts = Counter(
        "|".join(row.get(field) or [])
        for row in results
        if "error" not in row
    )
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="逐份直读官方PDF，用双期会计恒等式完成缺失财务事实统计"
    )
    argument_parser.add_argument("--workers", type=int, default=8)
    return argument_parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = _targets()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_target, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:  # 每份异常必须进入收据
                row = {
                    **{
                        key: value
                        for key, value in target.items()
                        if key != "document_sources"
                    },
                    "before_missing_metrics": [],
                    "after_missing_metrics": [],
                    "newly_admitted_metric_ids": [],
                    "document_complete_before_direct_statistics": False,
                    "document_complete_after_direct_statistics": False,
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            print(
                f"逐份PDF直接统计 {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"完整={row.get('document_complete_after_direct_statistics')}",
                flush=True,
            )
    results.sort(key=lambda item: item["announcement_id"])
    exception_count = sum("error" in row for row in results)
    complete_before = sum(
        bool(row.get("document_complete_before_direct_statistics"))
        for row in results
    )
    complete_after = sum(
        bool(row.get("document_complete_after_direct_statistics"))
        for row in results
    )
    receipt = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_DIRECT_PDF_STATISTICS_V1",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "base_parser_version": parser.PARSER_VERSION,
        "direct_parser_version": direct.DIRECT_PARSER_VERSION,
        "target_count": len(results),
        "complete_before_direct_statistics_count": complete_before,
        "complete_after_direct_statistics_count": complete_after,
        "newly_completed_document_count": complete_after - complete_before,
        "still_incomplete_count": len(results) - complete_after,
        "newly_admitted_metric_count": sum(
            len(row.get("newly_admitted_metric_ids") or []) for row in results
        ),
        "exception_count": exception_count,
        "before_missing_pattern_counts": _pattern_counts(
            results,
            "before_missing_metrics",
        ),
        "after_missing_pattern_counts": _pattern_counts(
            results,
            "after_missing_metrics",
        ),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        "逐份PDF直接统计完成："
        f"完整={complete_after}/{len(results)}；"
        f"新增完整文档={complete_after - complete_before}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"异常={exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
