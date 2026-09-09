from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_7 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)


PREVIOUS_REPLAY_RECEIPT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_6_gap_replay_sources/replay_results_v1_6/receipt.json"
)
GAP_SOURCE_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_7_gap_replay_sources"
)
GAP_SOURCE_MANIFEST = GAP_SOURCE_ROOT / "source_manifest.json"
V1_6_CHECKPOINT_ROOT = ROOT / (
    "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "checkpoints_v1_6"
)
OUTPUT_ROOT = GAP_SOURCE_ROOT / "replay_results_v1_7"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
NEGATIVE_GUARD_ID = "1204679732"

EXPECTED_RECOVERIES: dict[str, dict[str, str]] = {
    "1205243868": {"ACCOUNTS_RECEIVABLE_END": "20915754040"},
    "1202613205": {"CORE_PARENT_NET_PROFIT_YTD": "-9491000000"},
    "1202642581": {"CORE_PARENT_NET_PROFIT_YTD": "275000000"},
    "1202657301": {"CORE_PARENT_NET_PROFIT_YTD": "-4559830000"},
    "1202805449": {"CORE_PARENT_NET_PROFIT_YTD": "199926294.93"},
    "1203423195": {"CORE_PARENT_NET_PROFIT_YTD": "8997462.46"},
    "1202268047": {
        "OPERATING_PROFIT_YTD": "2461000000",
        "ACCOUNTS_RECEIVABLE_END": "2395000000",
        "INVENTORY_END": "2079000000",
        "TOTAL_LIABILITIES_END": "149348000000",
    },
    "1202532867": {"TOTAL_LIABILITIES_END": "201445532646"},
    "1202636509": {"PARENT_NET_PROFIT_YTD": "843459014.16"},
    "1202772943": {"PARENT_NET_PROFIT_YTD": "153065906.44"},
    "1211403108": {"PARENT_NET_PROFIT_YTD": "3558157949.42"},
}


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


def _v1_6_checkpoint(announcement_id: str) -> tuple[Path, dict[str, Any]]:
    matches = list(V1_6_CHECKPOINT_ROOT.rglob(f"{announcement_id}.json.gz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"V1.6基线检查点不是唯一一份：{announcement_id}|{len(matches)}"
        )
    with gzip.open(matches[0], "rt", encoding="utf-8") as handle:
        return matches[0], json.load(handle)


def _metric_values(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(metric["metric_id"]): str(metric["metric_value_cny"])
        for metric in payload.get("metrics") or []
    }


def _targets(only: set[str]) -> list[dict[str, Any]]:
    previous = json.loads(PREVIOUS_REPLAY_RECEIPT.read_text(encoding="utf-8"))
    targets: list[dict[str, Any]] = []
    for row in previous.get("results") or []:
        targets.append(
            {
                "announcement_id": str(row["announcement_id"]),
                "path": str(row["path"]),
                "expected_sha256": str(row["expected_sha256"]),
                "period_type": str(row["period_type"]),
                "target_class": "V1_6_AUTHORITATIVE_NONREGRESSION",
                "fix_family": "V1_6_AUTHORITATIVE_NONREGRESSION",
                "baseline_metric_values_cny": dict(row["metric_values_cny"]),
                "expected_metric_values_cny": dict(row["metric_values_cny"]),
                "baseline_missing_metrics": [],
            }
        )

    source = json.loads(GAP_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    for row in source.get("results") or []:
        announcement_id = str(row["announcement_id"])
        checkpoint_path, checkpoint = _v1_6_checkpoint(announcement_id)
        baseline = _metric_values(checkpoint)
        target_class = (
            "V1_7_NEGATIVE_GUARD"
            if announcement_id == NEGATIVE_GUARD_ID
            else "V1_7_POSITIVE_GAP"
        )
        expected = dict(baseline)
        if target_class == "V1_7_POSITIVE_GAP":
            if announcement_id not in EXPECTED_RECOVERIES:
                raise RuntimeError(f"V1.7正例缺少逐值契约：{announcement_id}")
            expected.update(EXPECTED_RECOVERIES[announcement_id])
            if set(expected) != set(REQUIRED_METRICS):
                raise RuntimeError(f"V1.7正例合并后不是九项：{announcement_id}")
        targets.append(
            {
                "announcement_id": announcement_id,
                "path": str(row["path"]),
                "expected_sha256": str(row["sha256"]),
                "period_type": str(row["period_type"]),
                "target_class": target_class,
                "fix_family": str(row["fix_family"]),
                "baseline_checkpoint_path": checkpoint_path.relative_to(
                    ROOT
                ).as_posix(),
                "baseline_checkpoint_sha256": _sha256(checkpoint_path.read_bytes()),
                "baseline_parser_version": str(checkpoint.get("parser_version") or ""),
                "baseline_metric_values_cny": baseline,
                "expected_metric_values_cny": expected,
                "baseline_missing_metrics": list(
                    checkpoint.get("missing_metrics") or []
                ),
            }
        )
    if only:
        targets = [row for row in targets if row["announcement_id"] in only]
    return targets


def _targeted_receipt_passed(row: dict[str, Any]) -> bool:
    announcement_id = str(row["announcement_id"])
    if row["target_class"] == "V1_6_AUTHORITATIVE_NONREGRESSION":
        return True
    if announcement_id == NEGATIVE_GUARD_ID:
        receipt = row.get("adjacent_note_receivable_receipt") or {}
        return (
            receipt.get("status")
            == "NO_MATCH_COMBINED_BALANCE_ROW_NOT_FOUND"
            and receipt.get("admitted_metric_ids") == []
            and row.get("document_complete") is False
            and row.get("missing_metrics") == ["ACCOUNTS_RECEIVABLE_END"]
        )
    if announcement_id == "1205243868":
        receipt = row.get("adjacent_note_receivable_receipt") or {}
        return (
            receipt.get("status")
            == "PASS_ADJACENT_NOTE_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED"
            and receipt.get("current_period_identity_passed") is True
            and receipt.get("prior_period_identity_passed") is True
        )
    if announcement_id in {
        "1202613205",
        "1202642581",
        "1202657301",
        "1202805449",
        "1203423195",
    }:
        receipt = row.get("extended_core_parent_profit_receipt") or {}
        return (
            receipt.get("status")
            == "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
            and receipt.get("admitted_metric_ids") == ["CORE_PARENT_NET_PROFIT_YTD"]
        )
    if announcement_id == "1202268047":
        receipt = row.get("quarterly_image_ocr_receipt") or {}
        return (
            receipt.get("status") == "PASS_QUARTERLY_IMAGE_OCR_METRICS_ADMITTED"
            and set(receipt.get("admitted_metric_ids") or [])
            == {
                "OPERATING_PROFIT_YTD",
                "ACCOUNTS_RECEIVABLE_END",
                "INVENTORY_END",
                "TOTAL_LIABILITIES_END",
            }
        )
    if announcement_id == "1202532867":
        receipt = row.get("dual_period_text_liability_receipt") or {}
        return (
            receipt.get("status")
            == "PASS_DUAL_PERIOD_TEXT_LIABILITY_IDENTITY_ADMITTED"
            and receipt.get("current_period_identity_passed") is True
            and receipt.get("prior_period_identity_passed") is True
        )
    if announcement_id in {"1202636509", "1202772943", "1211403108"}:
        receipt = row.get("extended_parent_net_profit_receipt") or {}
        return (
            receipt.get("status")
            == "PASS_EXTENDED_PARENT_NET_PROFIT_IDENTITY_ADMITTED"
            and receipt.get("current_period_identity_passed") is True
            and receipt.get("prior_period_identity_passed") is True
        )
    return False


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(target["path"])
    content = path.read_bytes()
    if _sha256(content) != target["expected_sha256"]:
        raise RuntimeError(f"固定PDF哈希漂移：{target['announcement_id']}")
    result = extract_official_pdf_facts(
        content,
        period_type=str(target["period_type"]),
    )
    observed = _metric_values(result)
    expected = {
        str(metric_id): Decimal(str(value))
        for metric_id, value in target["expected_metric_values_cny"].items()
    }
    observed_decimal = {
        metric_id: Decimal(value) for metric_id, value in observed.items()
    }
    baseline = {
        str(metric_id): Decimal(str(value))
        for metric_id, value in target["baseline_metric_values_cny"].items()
    }
    baseline_unchanged = all(observed_decimal.get(key) == value for key, value in baseline.items())
    row = {
        **target,
        "path": path.relative_to(ROOT).as_posix(),
        "parser_version": result.get("parser_version"),
        "document_complete": bool(result.get("document_complete")),
        "document_status": str(result.get("document_status")),
        "missing_metrics": list(result.get("missing_metrics") or []),
        "metric_count": len(observed_decimal),
        "metric_values_cny": observed,
        "expected_metric_values_match": observed_decimal == expected,
        "baseline_metric_values_unchanged": baseline_unchanged,
        "metric_evidence": list(result.get("metrics") or []),
        "extended_core_parent_profit_receipt": (
            result.get("v1_7_extended_core_parent_profit_receipt") or {}
        ),
        "extended_parent_net_profit_receipt": (
            result.get("v1_7_extended_parent_net_profit_receipt") or {}
        ),
        "adjacent_note_receivable_receipt": (
            result.get("v1_7_adjacent_note_receivable_reconciliation_receipt") or {}
        ),
        "dual_period_text_liability_receipt": (
            result.get("v1_7_dual_period_text_liability_receipt") or {}
        ),
        "quarterly_image_ocr_receipt": (
            result.get("v1_7_quarterly_image_ocr_receipt") or {}
        ),
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    row["targeted_receipt_passed"] = _targeted_receipt_passed(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="回放V1.6固定25份PDF、V1.7十一份正例及一份拒绝负例"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("并发数必须至少为1")
    targets = _targets(set(args.only))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_target, target): target for target in targets}
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as error:  # 每份异常必须落账，不得静默跳过
                row = {
                    **target,
                    "parser_version": None,
                    "document_complete": False,
                    "document_status": "NO_VIEW_V1_7_REPLAY_EXCEPTION",
                    "missing_metrics": list(REQUIRED_METRICS),
                    "metric_count": 0,
                    "metric_values_cny": {},
                    "expected_metric_values_match": False,
                    "baseline_metric_values_unchanged": False,
                    "targeted_receipt_passed": False,
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            results.append(row)
            print(
                f"V1.7回放 {completed}/{len(targets)}｜{row['announcement_id']}｜"
                f"{row.get('metric_count', 0)}/9｜逐值={row.get('expected_metric_values_match')}｜"
                f"基线不变={row.get('baseline_metric_values_unchanged')}｜"
                f"定向回执={row.get('targeted_receipt_passed')}",
                flush=True,
            )
    results.sort(key=lambda row: str(row["announcement_id"]))
    previous = [row for row in results if row["target_class"] == "V1_6_AUTHORITATIVE_NONREGRESSION"]
    positive = [row for row in results if row["target_class"] == "V1_7_POSITIVE_GAP"]
    negative = [row for row in results if row["target_class"] == "V1_7_NEGATIVE_GUARD"]
    value_match_count = sum(bool(row.get("expected_metric_values_match")) for row in results)
    baseline_unchanged_count = sum(bool(row.get("baseline_metric_values_unchanged")) for row in results)
    targeted_receipt_count = sum(bool(row.get("targeted_receipt_passed")) for row in results)
    passed = (
        len(results) == 37
        and len(previous) == 25
        and len(positive) == 11
        and len(negative) == 1
        and all(row.get("document_complete") is True for row in [*previous, *positive])
        and negative[0].get("document_complete") is False
        and negative[0].get("missing_metrics") == ["ACCOUNTS_RECEIVABLE_END"]
        and value_match_count == len(results)
        and baseline_unchanged_count == len(results)
        and targeted_receipt_count == len(results)
        and all(row.get("parser_version") == PARSER_VERSION for row in results)
    )
    receipt = {
        "protocol_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_7_REPLAY",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": PARSER_VERSION,
        "target_count": len(results),
        "v1_6_nonregression_target_count": len(previous),
        "v1_7_positive_gap_target_count": len(positive),
        "v1_7_negative_guard_target_count": len(negative),
        "complete_positive_and_nonregression_count": sum(
            bool(row.get("document_complete")) for row in [*previous, *positive]
        ),
        "expected_metric_values_match_count": value_match_count,
        "baseline_metric_values_unchanged_count": baseline_unchanged_count,
        "targeted_receipt_pass_count": targeted_receipt_count,
        "negative_guard_passed": bool(negative) and (
            negative[0].get("targeted_receipt_passed") is True
        ),
        "frozen_replay_contract_passed": passed,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        f"V1.7固定回放完成：前代非回归={len(previous)}；正例={len(positive)}；"
        f"拒绝负例={len(negative)}；总门={'通过' if passed else '未通过'}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
