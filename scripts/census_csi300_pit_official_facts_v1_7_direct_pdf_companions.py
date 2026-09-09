from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402


ALL_GAP_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_all_gap_census/receipt.json"
RAW_DAILY_ROOT = ROOT / (
    "data/raw/cninfo/periodic_report_metadata_v1_0_1/daily_checkpoints"
)
OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_direct_pdf_companions"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.cninfo.com.cn/",
}
PERIOD_TITLE_MARKERS = {
    "Q1": ("一季度", "第一季度"),
    "Q3": ("三季度", "第三季度"),
    "H1": ("半年", "中期"),
    "FY": ("年度报告", "年报"),
}
FORBIDDEN_COMPANION_TITLE_MARKERS = (
    "摘要",
    "英文",
    "取消",
)


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


def _same_day_records(publication_date: str) -> list[dict[str, Any]]:
    path = RAW_DAILY_ROOT / f"{publication_date}__combined.json.gz"
    if not path.exists():
        return []
    payload = _load_json_gzip(path)
    return list(payload.get("records") or [])


def _eligible_companions(
    row: dict[str, Any],
    checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    publication_date = str(checkpoint["event_publication_date"])[:10]
    same_company = [
        record
        for record in _same_day_records(publication_date)
        if str(record.get("secCode") or "") == str(checkpoint.get("sec_code") or "")
    ]
    current = next(
        (
            record
            for record in same_company
            if str(record.get("announcementId") or "") == row["announcement_id"]
        ),
        {},
    )
    current_size_kb = float(current.get("adjunctSize") or 0)
    current_title = str(checkpoint.get("announcement_title") or "")
    period_type = str(row["period_type"])
    markers = PERIOD_TITLE_MARKERS[period_type]
    candidates: list[dict[str, Any]] = []
    for record in same_company:
        announcement_id = str(record.get("announcementId") or "")
        if not announcement_id or announcement_id == row["announcement_id"]:
            continue
        title = str(record.get("announcementTitle") or "")
        if not any(marker in title for marker in markers):
            continue
        if any(marker in title for marker in FORBIDDEN_COMPANION_TITLE_MARKERS):
            continue
        relative_url = str(record.get("adjunctUrl") or "")
        if not relative_url.startswith("finalpage/") or not relative_url.endswith(
            (".PDF", ".pdf")
        ):
            continue
        size_kb = float(record.get("adjunctSize") or 0)
        is_explicit_full_text = "正文" in current_title and "全文" in title
        is_materially_larger = current_size_kb > 0 and size_kb > current_size_kb * 1.2
        if not is_explicit_full_text and not is_materially_larger:
            continue
        candidates.append(
            {
                "companion_announcement_id": announcement_id,
                "companion_announcement_title": title,
                "companion_relative_pdf_url": relative_url,
                "companion_official_pdf_url": (
                    f"https://static.cninfo.com.cn/{relative_url}"
                ),
                "companion_adjunct_size_kb": size_kb,
                "same_company": True,
                "same_period_type_from_title": True,
                "same_publication_date": True,
                "publication_date": publication_date,
                "selection_route": (
                    "SAME_DAY_EXPLICIT_FULL_TEXT_COMPANION"
                    if is_explicit_full_text
                    else "SAME_DAY_MATERIALLY_LARGER_PERIODIC_REPORT_COMPANION"
                ),
            }
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["companion_adjunct_size_kb"]),
            candidate["companion_announcement_id"],
        ),
    )


def _targets() -> list[dict[str, Any]]:
    all_gap = json.loads(ALL_GAP_RECEIPT_PATH.read_text(encoding="utf-8"))
    targets: list[dict[str, Any]] = []
    for row in all_gap["results"]:
        if not row.get("after_missing_metrics"):
            continue
        checkpoint_path = ROOT / row["checkpoint_path"]
        checkpoint = _load_json_gzip(checkpoint_path)
        candidates = _eligible_companions(row, checkpoint)
        if not candidates:
            continue
        targets.append(
            {
                "source_announcement_id": row["announcement_id"],
                "ts_code": row["ts_code"],
                "report_period": row["report_period"],
                "period_type": row["period_type"],
                "source_publication_date": str(
                    checkpoint["event_publication_date"]
                )[:10],
                "source_announcement_title": str(
                    checkpoint.get("announcement_title") or ""
                ),
                "source_checkpoint_path": row["checkpoint_path"],
                "source_checkpoint_sha256": row["checkpoint_sha256"],
                "source_pdf_sha256": row["expected_pdf_sha256"],
                "before_missing_metrics": list(row["after_missing_metrics"]),
                "v1_7_new_metrics_from_source_pdf": list(row.get("new_metrics") or []),
                "companion_candidates": candidates,
            }
        )
    return sorted(targets, key=lambda target: target["source_announcement_id"])


def _download(
    source_announcement_id: str,
    candidate: dict[str, Any],
) -> tuple[Path, bytes, str]:
    destination = OUTPUT_ROOT / (
        f"{source_announcement_id}__"
        f"{candidate['companion_announcement_id']}.pdf"
    )
    route = "REUSED_DIRECT_COMPANION_CACHE"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pdf.download")
        with requests.get(
            candidate["companion_official_pdf_url"],
            headers=HEADERS,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            total = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAXIMUM_PDF_BYTES:
                        raise RuntimeError(
                            "同日完整附件超过100 MiB："
                            f"{candidate['companion_announcement_id']}"
                        )
                    handle.write(chunk)
        temporary.replace(destination)
        route = "DOWNLOADED_FROM_CNINFO_STATIC_OFFICIAL_HOST"
    content = destination.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise RuntimeError(
            f"同日完整附件没有PDF文件头：{candidate['companion_announcement_id']}"
        )
    observed_kb = len(content) / 1024
    expected_kb = float(candidate["companion_adjunct_size_kb"])
    if expected_kb <= 0 or observed_kb <= 0:
        raise RuntimeError(
            f"同日完整附件大小无效：{candidate['companion_announcement_id']}"
        )
    return destination, content, route


def _metric_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(metric["metric_id"]): metric for metric in result.get("metrics") or []
    }


def _metric_value(metric: dict[str, Any]) -> Decimal:
    return Decimal(str(metric["metric_value_cny"]))


def _metric_tolerance(metric: dict[str, Any]) -> Decimal:
    unit = str(metric.get("source_unit") or "")
    if unit in parser._base.UNIT_MULTIPLIERS:
        return parser._base.UNIT_MULTIPLIERS[unit]
    multiplier = metric.get("source_unit_multiplier")
    if multiplier is not None:
        return Decimal(str(multiplier))
    return Decimal("1")


def _overlap_mismatches(
    baseline: dict[str, Any],
    companion: dict[str, Any],
) -> list[dict[str, Any]]:
    left = _metric_map(baseline)
    right = _metric_map(companion)
    mismatches: list[dict[str, Any]] = []
    for metric_id in sorted(set(left).intersection(right)):
        tolerance = max(_metric_tolerance(left[metric_id]), _metric_tolerance(right[metric_id]))
        difference = abs(_metric_value(left[metric_id]) - _metric_value(right[metric_id]))
        if difference <= tolerance:
            continue
        mismatches.append(
            {
                "metric_id": metric_id,
                "source_value_cny": str(_metric_value(left[metric_id])),
                "companion_value_cny": str(_metric_value(right[metric_id])),
                "absolute_difference_cny": str(difference),
                "allowed_display_tolerance_cny": str(tolerance),
            }
        )
    return mismatches


def _baseline_result(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["source_checkpoint_path"]
    if _sha256(checkpoint_path.read_bytes()) != target["source_checkpoint_sha256"]:
        raise RuntimeError(
            f"V1.6检查点哈希漂移：{target['source_announcement_id']}"
        )
    result = _load_json_gzip(checkpoint_path)
    for metric in target["v1_7_new_metrics_from_source_pdf"]:
        parser._v1_4._put_metric(
            result,
            parser._base.MetricEvidence(**metric),
        )
    parser._restamp_result(result)
    return result


def _merge_candidate(
    target: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    path, content, acquisition_route = _download(
        target["source_announcement_id"],
        candidate,
    )
    companion_result = parser.extract_official_pdf_facts(
        content,
        period_type=target["period_type"],
    )
    mismatches = _overlap_mismatches(baseline, companion_result)
    before_missing = list(baseline.get("missing_metrics") or [])
    merged = json.loads(json.dumps(baseline, ensure_ascii=False, allow_nan=False))
    admitted: list[str] = []
    if not mismatches:
        companion_metrics = _metric_map(companion_result)
        for metric_id in before_missing:
            metric = companion_metrics.get(metric_id)
            if metric is None:
                continue
            parser._v1_4._put_metric(
                merged,
                parser._base.MetricEvidence(**metric),
            )
            admitted.append(metric_id)
        parser._restamp_result(merged)
    return {
        **candidate,
        "path": path.relative_to(ROOT).as_posix(),
        "acquisition_route": acquisition_route,
        "observed_pdf_size_bytes": len(content),
        "observed_pdf_sha256": _sha256(content),
        "metadata_size_kb": candidate["companion_adjunct_size_kb"],
        "metadata_size_match_within_2kb": abs(
            len(content) / 1024 - float(candidate["companion_adjunct_size_kb"])
        )
        <= 2,
        "companion_parser_version": companion_result.get("parser_version"),
        "companion_document_complete": bool(
            companion_result.get("document_complete")
        ),
        "companion_missing_metrics": list(
            companion_result.get("missing_metrics") or []
        ),
        "overlap_metric_count": len(
            set(_metric_map(baseline)).intersection(_metric_map(companion_result))
        ),
        "overlap_mismatches": mismatches,
        "newly_admitted_metric_ids": admitted,
        "new_metrics": [
            metric
            for metric in merged.get("metrics") or []
            if metric.get("metric_id") in admitted
        ],
        "after_missing_metrics": list(merged.get("missing_metrics") or []),
        "document_complete_after_companion": bool(merged.get("document_complete")),
    }


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    baseline = _baseline_result(target)
    candidate_results: list[dict[str, Any]] = []
    for candidate in target["companion_candidates"]:
        try:
            candidate_results.append(
                _merge_candidate(target, baseline, candidate)
            )
        except Exception as error:  # 每个候选失败单独留痕，不吞掉其他候选
            candidate_results.append(
                {
                    **candidate,
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "newly_admitted_metric_ids": [],
                    "after_missing_metrics": list(
                        baseline.get("missing_metrics") or []
                    ),
                    "document_complete_after_companion": False,
                }
            )
    ranked = sorted(
        candidate_results,
        key=lambda row: (
            -len(row.get("newly_admitted_metric_ids") or []),
            len(row.get("after_missing_metrics") or []),
            str(row.get("companion_announcement_id") or ""),
        ),
    )
    selected = ranked[0]
    return {
        **{key: value for key, value in target.items() if key != "companion_candidates"},
        "candidate_count": len(candidate_results),
        "candidate_results": candidate_results,
        "selected_companion_announcement_id": selected.get(
            "companion_announcement_id"
        ),
        "selected_companion_announcement_title": selected.get(
            "companion_announcement_title"
        ),
        "selected_companion_official_pdf_url": selected.get(
            "companion_official_pdf_url"
        ),
        "selected_companion_pdf_sha256": selected.get("observed_pdf_sha256"),
        "selected_newly_admitted_metric_ids": list(
            selected.get("newly_admitted_metric_ids") or []
        ),
        "selected_new_metrics": list(selected.get("new_metrics") or []),
        "after_missing_metrics": list(selected.get("after_missing_metrics") or []),
        "document_complete_after_companion": bool(
            selected.get("document_complete_after_companion")
        ),
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="逐份读取同日官方全文附件并交叉核对缺失财务事实"
    )
    argument_parser.add_argument("--workers", type=int, default=6)
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
            except Exception as error:  # 单份异常必须进入最终收据
                row = {
                    **{
                        key: value
                        for key, value in target.items()
                        if key != "companion_candidates"
                    },
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "selected_newly_admitted_metric_ids": [],
                    "after_missing_metrics": list(
                        target.get("before_missing_metrics") or []
                    ),
                    "document_complete_after_companion": False,
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            print(
                f"同日官方全文逐份结算 {completed}/{len(targets)}｜"
                f"{row['source_announcement_id']}｜"
                f"新增={len(row.get('selected_newly_admitted_metric_ids') or [])}｜"
                f"完整={row.get('document_complete_after_companion')}",
                flush=True,
            )
    results.sort(key=lambda row: row["source_announcement_id"])
    status_counts = Counter(
        "COMPLETE"
        if row.get("document_complete_after_companion")
        else "INCOMPLETE"
        for row in results
    )
    receipt = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_7_DIRECT_PDF_COMPANION_CENSUS",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": parser.PARSER_VERSION,
        "target_source_document_count": len(results),
        "complete_after_companion_count": status_counts["COMPLETE"],
        "still_incomplete_after_companion_count": status_counts["INCOMPLETE"],
        "newly_admitted_metric_count": sum(
            len(row.get("selected_newly_admitted_metric_ids") or [])
            for row in results
        ),
        "exception_count": sum("error" in row for row in results),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        "同日官方全文逐份结算完成："
        f"完整={receipt['complete_after_companion_count']}/{len(results)}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"异常={receipt['exception_count']}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if receipt["exception_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
