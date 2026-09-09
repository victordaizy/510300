from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_5 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
    quarterly_contamination_corrections,
)
from scripts import replay_csi300_pit_official_facts_v1_4 as _v1_4_replay  # noqa: E402


SOURCE_ROOT = _v1_4_replay.SOURCE_ROOT
REFERENCE_RECEIPT_PATH = SOURCE_ROOT / "replay_results/receipt.json"
OUTPUT_ROOT = SOURCE_ROOT / "replay_results_v1_5"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(target["path"]))
    content = path.read_bytes()
    observed_sha256 = _v1_4_replay._sha256(content)
    if observed_sha256 != target["expected_sha256"]:
        raise ValueError(f"固定PDF哈希漂移：{target['announcement_id']}")
    result = extract_official_pdf_facts(
        content,
        period_type=str(target["period_type"]),
    )
    if result.get("parser_version") != PARSER_VERSION:
        raise RuntimeError(f"解析器版本回执不一致：{target['announcement_id']}")
    metrics = {str(metric["metric_id"]): metric for metric in result.get("metrics") or []}
    return {
        **target,
        "path": path.relative_to(ROOT).as_posix(),
        "document_complete": bool(result["document_complete"]),
        "document_status": str(result["document_status"]),
        "missing_metrics": list(result["missing_metrics"]),
        "metric_count": len(metrics),
        "metric_values_cny": {
            metric_id: metrics[metric_id]["metric_value_cny"]
            for metric_id in REQUIRED_METRICS
            if metric_id in metrics
        },
        "metric_evidence": [metrics[key] for key in REQUIRED_METRICS if key in metrics],
        "sections": result.get("sections") or {},
        "deferred_expensive_paths_receipt": (
            result.get("v1_5_deferred_expensive_paths_receipt") or {}
        ),
        "hash_bound_manual_appendix_receipt": (
            result.get("hash_bound_manual_appendix_receipt") or {}
        ),
        "quarterly_contamination_guard_receipt": (
            result.get("v1_5_quarterly_contamination_guard_receipt") or {}
        ),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放固定22份PDF并验证V1.5延迟昂贵路径")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", action="append", default=[])
    return parser.parse_args()


def _reference_values() -> dict[str, dict[str, Decimal]]:
    payload = json.loads(REFERENCE_RECEIPT_PATH.read_text(encoding="utf-8"))
    return {
        str(row["announcement_id"]): {
            str(metric_id): Decimal(str(value))
            for metric_id, value in (row.get("metric_values_cny") or {}).items()
        }
        for row in payload.get("results") or []
    }


def _authoritative_values(
    references: dict[str, dict[str, Decimal]],
) -> tuple[dict[str, dict[str, Decimal]], dict[str, set[str]]]:
    authoritative = {
        announcement_id: dict(metric_values)
        for announcement_id, metric_values in references.items()
    }
    corrected_metric_ids: dict[str, set[str]] = {}
    for pdf_sha256, document in quarterly_contamination_corrections().items():
        announcement_id = str(document["announcement_id"])
        if announcement_id not in authoritative:
            raise RuntimeError(
                f"季度表污染修正公告不在V1.4固定回放中：{announcement_id}"
            )
        corrected_metric_ids[announcement_id] = set()
        for metric_id, correction in document["corrected_metrics"].items():
            frozen_value = authoritative[announcement_id].get(str(metric_id))
            declared_frozen_value = Decimal(
                str(correction["v1_4_incorrect_value_cny"])
            )
            if frozen_value != declared_frozen_value:
                raise RuntimeError(
                    f"季度表污染修正的V1.4旧值不一致：{announcement_id}|"
                    f"{metric_id}|{frozen_value}|{declared_frozen_value}"
                )
            authoritative[announcement_id][str(metric_id)] = Decimal(
                str(correction["corrected_value_cny"])
            )
            corrected_metric_ids[announcement_id].add(str(metric_id))
        if not str(pdf_sha256):
            raise RuntimeError("季度表污染修正PDF哈希为空")
    return authoritative, corrected_metric_ids


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    targets = _v1_4_replay._targets(set(args.only))
    references = _reference_values()
    authoritative, corrected_metric_ids = _authoritative_values(references)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_target, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - 每份失败必须落账
                result = {
                    **target,
                    "path": Path(str(target["path"])).relative_to(ROOT).as_posix(),
                    "document_complete": False,
                    "document_status": "NO_VIEW_V1_5_REPLAY_EXCEPTION",
                    "missing_metrics": list(REQUIRED_METRICS),
                    "metric_count": 0,
                    "metric_values_cny": {},
                    "metric_evidence": [],
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            observed = {
                str(metric_id): Decimal(str(value))
                for metric_id, value in (result.get("metric_values_cny") or {}).items()
            }
            announcement_id = str(result["announcement_id"])
            frozen_reference = references.get(announcement_id, {})
            result["v1_4_reference_metric_values_match"] = observed == frozen_reference
            result["authoritative_metric_values_match"] = (
                observed == authoritative.get(announcement_id, {})
            )
            result["metric_differences_from_v1_4"] = {
                metric_id: {
                    "v1_4_value_cny": str(frozen_reference.get(metric_id)),
                    "v1_5_value_cny": str(observed.get(metric_id)),
                    "registered_quarterly_contamination_correction": metric_id
                    in corrected_metric_ids.get(announcement_id, set()),
                }
                for metric_id in REQUIRED_METRICS
                if observed.get(metric_id) != frozen_reference.get(metric_id)
            }
            results.append(result)
            print(
                f"V1.5回放 {completed}/{len(targets)}｜{result['announcement_id']}｜"
                f"{result['metric_count']}/9｜权威逐值一致="
                f"{result['authoritative_metric_values_match']}",
                flush=True,
            )
    results.sort(key=lambda item: str(item["announcement_id"]))
    complete_count = sum(bool(item["document_complete"]) for item in results)
    exact_match_count = sum(
        bool(item["v1_4_reference_metric_values_match"]) for item in results
    )
    authoritative_match_count = sum(
        bool(item["authoritative_metric_values_match"]) for item in results
    )
    correction_document_count = sum(
        bool(item.get("metric_differences_from_v1_4")) for item in results
    )
    correction_metric_count = sum(
        len(item.get("metric_differences_from_v1_4") or {}) for item in results
    )
    full_path_count = sum(
        bool(
            (item.get("deferred_expensive_paths_receipt") or {}).get(
                "full_v1_4_path_executed"
            )
        )
        for item in results
    )
    expected_correction_document_count = len(corrected_metric_ids)
    expected_correction_metric_count = sum(map(len, corrected_metric_ids.values()))
    passed = (
        complete_count == len(results)
        and authoritative_match_count == len(results)
        and correction_document_count == expected_correction_document_count
        and correction_metric_count == expected_correction_metric_count
    )
    receipt = {
        "protocol_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_5_REPLAY",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": PARSER_VERSION,
        "target_count": len(results),
        "complete_count": complete_count,
        "incomplete_count": len(results) - complete_count,
        "exact_v1_4_metric_value_match_count": exact_match_count,
        "authoritative_metric_value_match_count": authoritative_match_count,
        "quarterly_contamination_corrected_document_count": correction_document_count,
        "quarterly_contamination_corrected_metric_count": correction_metric_count,
        "full_v1_4_path_executed_count": full_path_count,
        "all_nine_metric_replay_and_authoritative_value_match_passed": passed,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        f"V1.5固定回放完成：完整={complete_count}/{len(results)}；"
        f"权威逐值一致={authoritative_match_count}/{len(results)}；"
        f"修正季度污染={correction_document_count}份/{correction_metric_count}项；"
        f"完整V1.4慢路径={full_path_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
