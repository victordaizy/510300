from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_residual_v1 as residual  # noqa: E402
from scripts import census_csi300_pit_official_facts_v1_7_residual_direct_pdf_statistics as census  # noqa: E402


PROTOCOL_ID = "CSI300_PIT_OFFICIAL_FACTS_RESIDUAL_DIRECT_PDF_FINALIZE_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
FULL_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_7_residual_direct_pdf_statistics_v1_1/receipt.json"
)
OVERRIDE_RECEIPT_PATHS = (
    ROOT / "tmp/pit_v1_7_residual_final_smoke/receipt.json",
)
OUTPUT_RECEIPT_PATH = (
    ROOT / "tmp/pit_v1_7_residual_direct_pdf_statistics_final/receipt.json"
)


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    content = path.read_bytes()
    return json.loads(content.decode("utf-8")), content


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    full_receipt, full_content = _load(FULL_RECEIPT_PATH)
    if (
        full_receipt.get("target_exception_count") != 0
        or full_receipt.get("source_exception_count") != 0
        or full_receipt.get("market_price_read") is not False
        or full_receipt.get("future_return_read") is not False
    ):
        raise RuntimeError("全量残缺直读收据不满足合成门槛")

    result_map = {
        str(row["announcement_id"]): row
        for row in full_receipt.get("results") or []
    }
    override_inputs: list[dict[str, Any]] = []
    overridden_ids: list[str] = []
    for path in OVERRIDE_RECEIPT_PATHS:
        receipt, content = _load(path)
        if (
            receipt.get("target_exception_count") != 0
            or receipt.get("source_exception_count") != 0
            or receipt.get("market_price_read") is not False
            or receipt.get("future_return_read") is not False
        ):
            raise RuntimeError(f"覆盖收据不满足合成门槛：{_relative(path)}")
        for row in receipt.get("results") or []:
            announcement_id = str(row["announcement_id"])
            if announcement_id not in result_map:
                raise RuntimeError(f"覆盖公告不在全量目标中：{announcement_id}")
            result_map[announcement_id] = row
            overridden_ids.append(announcement_id)
        override_inputs.append(
            {
                "path": _relative(path),
                "sha256": census._sha256(content),
                "announcement_ids": sorted(
                    str(row["announcement_id"])
                    for row in receipt.get("results") or []
                ),
            }
        )

    results = sorted(result_map.values(), key=lambda row: str(row["announcement_id"]))
    manifest = pd.read_parquet(census.PATCH_MANIFEST_PATH)
    manifest_rows = manifest.to_dict("records")
    combined_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        announcement_id = str(manifest_row["announcement_id"])
        result = result_map.get(announcement_id)
        final_missing = json.loads(str(manifest_row["final_missing_metrics_json"]))
        combined_rows.append(
            {
                "announcement_id": announcement_id,
                "period_type": str(manifest_row["period_type"]),
                "after_missing_metrics": list(
                    result.get("after_missing_metrics")
                    if result is not None and "error" not in result
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
        **{
            key: value
            for key, value in full_receipt.items()
            if key
            not in {
                "created_at",
                "results",
                "composition",
            }
        },
        "protocol_id": PROTOCOL_ID,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "residual_parser_version": residual.RESIDUAL_PARSER_VERSION,
        "explicit_announcement_ids": [],
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
        "composition": {
            "input_full_receipt_path": _relative(FULL_RECEIPT_PATH),
            "input_full_receipt_sha256": census._sha256(full_content),
            "override_receipts": override_inputs,
            "overridden_announcement_ids": sorted(set(overridden_ids)),
        },
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    census._atomic_write_json(OUTPUT_RECEIPT_PATH, receipt)
    print(
        "残缺直读最终收据完成："
        f"目标={len(results)}；"
        f"全体完整={overall_complete_after}/{len(combined_rows)}；"
        f"新增指标={receipt['newly_admitted_metric_count']}；"
        f"修正指标={receipt['corrected_metric_count']}；"
        f"未替换撤回={receipt['removed_unreplaced_metric_count']}；"
        f"异常={target_exception_count + source_exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
