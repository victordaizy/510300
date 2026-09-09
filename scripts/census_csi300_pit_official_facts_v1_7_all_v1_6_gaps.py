from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from zoneinfo import ZoneInfo

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser  # noqa: E402


CHECKPOINT_ROOT = ROOT / (
    "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "checkpoints_v1_6"
)
OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_all_gap_census"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.cninfo.com.cn/",
}
CACHE_ROOTS = (
    ROOT / "tmp/pit_v1_7_core_gap_census",
    ROOT / "tmp/pit_v1_7_core_diagnostics",
    ROOT / "tmp/pit_v1_7_diagnostics",
    ROOT / "tmp/pit_v1_7_image4_diagnostics",
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


def _targets() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for checkpoint_path in CHECKPOINT_ROOT.rglob("*.json.gz"):
        with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if checkpoint.get("document_terminal") is not True:
            continue
        if checkpoint.get("document_complete") is True:
            continue
        required = {
            "announcement_id": str(checkpoint["announcement_id"]),
            "ts_code": str(checkpoint["ts_code"]),
            "report_period": str(checkpoint["report_period"]),
            "period_type": str(checkpoint["period_type"]),
            "official_pdf_url": str(checkpoint["official_pdf_url"]),
            "expected_pdf_sha256": str(checkpoint["official_pdf_sha256"]),
            "expected_pdf_size_bytes": int(checkpoint["official_pdf_size_bytes"]),
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": _sha256(checkpoint_path.read_bytes()),
            "before_missing_metrics": list(checkpoint.get("missing_metrics") or []),
        }
        if not required["official_pdf_url"].startswith(
            "https://static.cninfo.com.cn/"
        ):
            raise RuntimeError(
                f"非固定官方主机：{required['announcement_id']}|"
                f"{required['official_pdf_url']}"
            )
        results.append(required)
    return sorted(results, key=lambda row: row["announcement_id"])


def _cache_candidates(announcement_id: str) -> list[Path]:
    names = (
        f"{announcement_id}.pdf",
        f"{announcement_id}.PDF",
    )
    return [root / name for root in CACHE_ROOTS for name in names]


def _download(target: dict[str, Any]) -> tuple[Path, str]:
    destination = OUTPUT_ROOT / f"{target['announcement_id']}.pdf"
    route = "REUSED_HASH_VERIFIED_ALL_GAP_CACHE"
    if not destination.exists():
        for candidate in _cache_candidates(target["announcement_id"]):
            if not candidate.exists():
                continue
            candidate_content = candidate.read_bytes()
            if (
                len(candidate_content) == target["expected_pdf_size_bytes"]
                and _sha256(candidate_content) == target["expected_pdf_sha256"]
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate, destination)
                route = "COPIED_FROM_HASH_VERIFIED_V1_7_CACHE"
                break
    if not destination.exists():
        temporary = destination.with_suffix(".pdf.download")
        with requests.get(
            target["official_pdf_url"],
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
                            f"官方PDF超过100 MiB：{target['announcement_id']}"
                        )
                    handle.write(chunk)
        temporary.replace(destination)
        route = "DOWNLOADED_FROM_CHECKPOINT_OFFICIAL_URL"
    content = destination.read_bytes()
    checks = {
        "pdf_header": content.startswith(b"%PDF-"),
        "size": len(content) == target["expected_pdf_size_bytes"],
        "sha256": _sha256(content) == target["expected_pdf_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"官方PDF与V1.6检查点不一致：{target['announcement_id']}|{failed}"
        )
    return destination, route


def _load_checkpoint(target: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = ROOT / target["checkpoint_path"]
    if _sha256(checkpoint_path.read_bytes()) != target["checkpoint_sha256"]:
        raise RuntimeError(f"V1.6检查点哈希漂移：{target['announcement_id']}")
    with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path, acquisition_route = _download(target)
    content = path.read_bytes()
    result = _load_checkpoint(target)
    before_missing = tuple(result.get("missing_metrics") or [])
    page_texts, text_receipt = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(text) for text in page_texts
    ]

    repair_receipts: dict[str, Any] = {}
    repair_receipts["extended_parent_net_profit"] = (
        parser._add_extended_parent_net_profit(
            result,
            normalized,
            period_type=target["period_type"],
        )
    )
    repair_receipts["extended_core_parent_profit"] = (
        parser._add_extended_core_parent_profit(
            result,
            normalized,
            period_type=target["period_type"],
        )
    )
    repair_receipts["summary_reconciled_generic_statement"] = (
        parser._add_summary_reconciled_generic_statement_metrics(
            result,
            normalized,
            period_type=target["period_type"],
        )
    )
    repair_receipts["adjacent_note_receivable_reconciliation"] = (
        parser._add_adjacent_note_net_receivable_reconciliation(result, normalized)
    )
    repair_receipts["dual_period_text_liability"] = (
        parser._add_dual_period_text_liability(result, normalized)
    )
    repair_receipts["quarterly_image_ocr"] = parser._add_quarterly_image_ocr(
        result,
        content,
        normalized,
        period_type=target["period_type"],
    )
    parser._restamp_result(result)

    after_missing = tuple(result.get("missing_metrics") or [])
    newly_admitted_ids = [
        metric_id for metric_id in before_missing if metric_id not in after_missing
    ]
    new_metrics = [
        metric
        for metric in result.get("metrics") or []
        if metric.get("metric_id") in newly_admitted_ids
    ]
    return {
        **target,
        "path": path.relative_to(ROOT).as_posix(),
        "acquisition_route": acquisition_route,
        "observed_pdf_sha256": _sha256(content),
        "observed_pdf_size_bytes": len(content),
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
        "after_missing_metrics": list(after_missing),
        "newly_admitted_metric_ids": newly_admitted_ids,
        "new_metrics": new_metrics,
        "document_complete_after_repair": len(after_missing) == 0,
        "repair_receipts": repair_receipts,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="对V1.6全部不完整官方PDF运行V1.7定向修复普查"
    )
    argument_parser.add_argument("--workers", type=int, default=4)
    return argument_parser.parse_args()


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
            except Exception as error:  # 每份失败必须进入收据
                row = {
                    **target,
                    "document_complete_after_repair": False,
                    "newly_admitted_metric_ids": [],
                    "repair_status": "NO_VIEW_V1_7_ALL_GAP_CENSUS_EXCEPTION",
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            print(
                f"V1.7全缺口普查 {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"完整={row.get('document_complete_after_repair')}",
                flush=True,
            )
    results.sort(key=lambda row: row["announcement_id"])
    exception_count = sum("error" in row for row in results)
    complete_count = sum(
        bool(row.get("document_complete_after_repair")) for row in results
    )
    newly_admitted_metric_count = sum(
        len(row.get("newly_admitted_metric_ids") or []) for row in results
    )
    receipt = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_7_ALL_V1_6_GAP_CENSUS",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": parser.PARSER_VERSION,
        "target_count": len(results),
        "complete_after_repair_count": complete_count,
        "still_incomplete_count": len(results) - complete_count,
        "newly_admitted_metric_count": newly_admitted_metric_count,
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
        f"V1.7全缺口普查完成：完整={complete_count}/{len(results)}；"
        f"新增指标={newly_admitted_metric_count}；异常={exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
