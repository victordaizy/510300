from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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


from scripts import census_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as census  # noqa: E402
from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_residual_v1 as residual  # noqa: E402


CURRENT_QUEUE_PATH = ROOT / (
    "data/audit/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "official_fact_document_queue_v1_7_residual_direct_pdf.parquet"
)
DIRECT_RECEIPT_PATH = ROOT / "tmp/pit_v1_7_direct_pdf_statistics/receipt.json"
DEFAULT_OUTPUT_ROOT = ROOT / "tmp/pit_v1_8_current_residual_direct_pdf_statistics"
TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_ID = (
    "CSI300_PIT_OFFICIAL_FACTS_CURRENT_RESIDUAL_DIRECT_PDF_CENSUS_V1_8"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _resolve_output_root(value: str) -> Path:
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


def _build_targets(
    queue_rows: list[dict[str, Any]],
    direct_map: dict[str, dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        announcement_id = str(queue_row["announcement_id"])
        missing = _missing_metrics(queue_row)
        if not missing:
            continue
        if selected_ids and announcement_id not in selected_ids:
            continue
        direct_row = direct_map.get(announcement_id)
        if direct_row is None:
            raise ValueError(f"直读来源收据缺少公告：{announcement_id}")
        sources = [
            contract
            for source in direct_row.get("source_results") or []
            if (contract := census._source_contract(source)) is not None
        ]
        if not sources:
            raise ValueError(f"公告没有可复用的官方PDF来源：{announcement_id}")
        targets.append(
            {
                "announcement_id": announcement_id,
                "ts_code": str(queue_row["ts_code"]),
                "report_period": str(queue_row["report_period"]),
                "period_type": str(queue_row["period_type"]),
                "merged_checkpoint_path": str(queue_row["checkpoint_path"]),
                "merged_checkpoint_sha256": str(queue_row["checkpoint_sha256"]),
                "document_sources": sources,
            }
        )
    targets.sort(key=lambda item: item["announcement_id"])
    if selected_ids:
        found_ids = {target["announcement_id"] for target in targets}
        unavailable = sorted(selected_ids - found_ids)
        if unavailable:
            raise ValueError(
                "指定公告不在当前残缺集合：" + ",".join(unavailable)
            )
    return targets


def _error_row(target: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
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


def _combined_rows(
    queue_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result_map = {str(row["announcement_id"]): row for row in results}
    combined: list[dict[str, Any]] = []
    for queue_row in queue_rows:
        announcement_id = str(queue_row["announcement_id"])
        result_row = result_map.get(announcement_id)
        missing = _missing_metrics(queue_row)
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
        description=(
            "仅对最新队列中的残缺文档逐份直读官方PDF，跳过所有已完整文档"
        )
    )
    argument_parser.add_argument("--workers", type=int, default=6)
    argument_parser.add_argument(
        "--announcement-id",
        action="append",
        default=[],
        help="仅处理指定的当前残缺公告；可重复传入",
    )
    argument_parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT.relative_to(ROOT)),
    )
    argument_parser.add_argument(
        "--input-queue",
        default=str(CURRENT_QUEUE_PATH.relative_to(ROOT)),
        help="作为本轮起点的最新文档队列 Parquet",
    )
    return argument_parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    output_root = _resolve_output_root(args.output_root)
    receipt_path = output_root / "receipt.json"
    output_root.mkdir(parents=True, exist_ok=True)

    input_queue_path = _resolve_output_root(args.input_queue)
    queue_content = input_queue_path.read_bytes()
    queue = pd.read_parquet(input_queue_path)
    queue_rows = queue.to_dict(orient="records")
    direct_content = DIRECT_RECEIPT_PATH.read_bytes()
    direct_receipt = json.loads(direct_content.decode("utf-8"))
    direct_map = {
        str(row["announcement_id"]): row
        for row in direct_receipt.get("results") or []
    }
    selected_ids = {str(value) for value in args.announcement_id}
    targets = _build_targets(queue_rows, direct_map, selected_ids)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(census._run_target, target): target
            for target in targets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = _error_row(target, error)
            results.append(row)
            print(
                f"最新残缺直读 {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜"
                f"新增={len(row.get('newly_admitted_metric_ids') or [])}｜"
                f"修正={len(row.get('corrected_metric_ids') or [])}｜"
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
        "input_current_queue_path": str(input_queue_path.relative_to(ROOT)),
        "input_current_queue_sha256": _sha256(queue_content),
        "input_direct_receipt_path": str(DIRECT_RECEIPT_PATH.relative_to(ROOT)),
        "input_direct_receipt_sha256": _sha256(direct_content),
        "residual_parser_version": residual.RESIDUAL_PARSER_VERSION,
        "worker_count": args.workers,
        "explicit_announcement_ids": sorted(selected_ids),
        "input_document_count": len(queue_rows),
        "input_incomplete_document_count": sum(
            bool(_missing_metrics(row)) for row in queue_rows
        ),
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
        "before_missing_pattern_counts": census._pattern_counts(
            results,
            "before_missing_metrics",
        ),
        "after_missing_pattern_counts": census._pattern_counts(
            results,
            "after_missing_metrics",
        ),
        "newly_admitted_metric_counts": census._metric_counts(
            results,
            "newly_admitted_metric_ids",
        ),
        "corrected_metric_counts": census._metric_counts(
            results,
            "corrected_metric_ids",
        ),
        "overall_residual_missing_pattern_counts": census._pattern_counts(
            combined_rows,
            "after_missing_metrics",
        ),
        "overall_residual_missing_metric_counts": census._metric_counts(
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
    census._atomic_write_json(receipt_path, receipt)
    print(
        "最新残缺直读完成："
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
    return 0 if target_exception_count == 0 and source_exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
