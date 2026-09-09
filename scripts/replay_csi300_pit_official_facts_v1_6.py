from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_6 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)


LEGACY_REPLAY_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_4_replay_sources"
)
V1_5_RECEIPT_PATH = LEGACY_REPLAY_ROOT / "replay_results_v1_5/receipt.json"
GAP_SOURCE_ROOT = ROOT / (
    "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_6_gap_replay_sources"
)
GAP_SOURCE_MANIFEST_PATH = GAP_SOURCE_ROOT / "source_manifest.json"
OUTPUT_ROOT = GAP_SOURCE_ROOT / "replay_results_v1_6"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")

GAP_EXPECTED_VALUES: dict[str, dict[str, str]] = {
    "1202600713": {
        "OPERATING_REVENUE_YTD": "23973109368",
        "OPERATING_PROFIT_YTD": "3967758600",
        "PARENT_NET_PROFIT_YTD": "3354918405",
        "CORE_PARENT_NET_PROFIT_YTD": "2992606000",
        "OPERATING_CASH_FLOW_YTD": "4667287024",
        "ACCOUNTS_RECEIVABLE_END": "613179913",
        "INVENTORY_END": "4181176707",
        "TOTAL_ASSETS_END": "103123000816",
        "TOTAL_LIABILITIES_END": "28024570333",
    },
    "1205362872": {
        "OPERATING_REVENUE_YTD": "39433777000",
        "OPERATING_PROFIT_YTD": "-1775911000",
        "PARENT_NET_PROFIT_YTD": "-7824190000",
        "CORE_PARENT_NET_PROFIT_YTD": "-2379203000",
        "OPERATING_CASH_FLOW_YTD": "-5046386000",
        "ACCOUNTS_RECEIVABLE_END": "20791301000",
        "INVENTORY_END": "26316928000",
        "TOTAL_ASSETS_END": "120709368000",
        "TOTAL_LIABILITIES_END": "87001180000",
    },
    "1203181067": {
        "OPERATING_REVENUE_YTD": "32419339546.28",
        "OPERATING_PROFIT_YTD": "780649747.88",
        "PARENT_NET_PROFIT_YTD": "830337431.10",
        "CORE_PARENT_NET_PROFIT_YTD": "519613629.14",
        "OPERATING_CASH_FLOW_YTD": "3096115453.01",
        "ACCOUNTS_RECEIVABLE_END": "663524805.12",
        "INVENTORY_END": "1949548273.48",
        "TOTAL_ASSETS_END": "27534301433.55",
        "TOTAL_LIABILITIES_END": "14320872456.53",
    },
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _targets(only: set[str]) -> list[dict[str, Any]]:
    v1_5 = json.loads(V1_5_RECEIPT_PATH.read_text(encoding="utf-8"))
    legacy = []
    for row in v1_5.get("results") or []:
        announcement_id = str(row["announcement_id"])
        legacy.append(
            {
                "announcement_id": announcement_id,
                "path": str(row["path"]),
                "expected_sha256": str(row["expected_sha256"]),
                "period_type": str(row["period_type"]),
                "fix_family": "V1_5_AUTHORITATIVE_NONREGRESSION",
                "expected_metric_values_cny": {
                    str(metric_id): str(value)
                    for metric_id, value in (row.get("metric_values_cny") or {}).items()
                },
            }
        )

    source = json.loads(GAP_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    gap = []
    for row in source.get("results") or []:
        announcement_id = str(row["announcement_id"])
        if announcement_id not in GAP_EXPECTED_VALUES:
            raise RuntimeError(f"V1.6缺口样本没有权威逐值契约：{announcement_id}")
        gap.append(
            {
                "announcement_id": announcement_id,
                "path": str(row["path"]),
                "expected_sha256": str(row["sha256"]),
                "period_type": str(row["period_type"]),
                "fix_family": str(row["fix_family"]),
                "expected_metric_values_cny": GAP_EXPECTED_VALUES[announcement_id],
            }
        )
    targets = [*legacy, *gap]
    if only:
        targets = [row for row in targets if row["announcement_id"] in only]
    return targets


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / str(target["path"])
    content = path.read_bytes()
    if _sha256(content) != target["expected_sha256"]:
        raise RuntimeError(f"固定PDF哈希漂移：{target['announcement_id']}")
    result = extract_official_pdf_facts(
        content,
        period_type=str(target["period_type"]),
    )
    observed = {
        str(metric["metric_id"]): str(metric["metric_value_cny"])
        for metric in result.get("metrics") or []
    }
    expected = {
        str(metric_id): Decimal(str(value))
        for metric_id, value in target["expected_metric_values_cny"].items()
    }
    observed_decimal = {
        metric_id: Decimal(value)
        for metric_id, value in observed.items()
    }
    return {
        **target,
        "path": path.relative_to(ROOT).as_posix(),
        "parser_version": result.get("parser_version"),
        "document_complete": bool(result.get("document_complete")),
        "document_status": str(result.get("document_status")),
        "missing_metrics": list(result.get("missing_metrics") or []),
        "metric_count": len(observed_decimal),
        "metric_values_cny": observed,
        "authoritative_metric_values_match": observed_decimal == expected,
        "metric_evidence": list(result.get("metrics") or []),
        "inverse_core_label_receipt": result.get("v1_6_inverse_core_label_receipt") or {},
        "net_receivable_reconciliation_receipt": (
            result.get("v1_6_net_receivable_reconciliation_receipt") or {}
        ),
        "anchored_image_ocr_receipt": (
            result.get("v1_6_anchored_image_ocr_receipt") or {}
        ),
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def _targeted_receipt_passed(row: dict[str, Any]) -> bool:
    announcement_id = str(row["announcement_id"])
    if announcement_id == "1202600713":
        receipt = row.get("inverse_core_label_receipt") or {}
        return (
            receipt.get("status") == "PASS_EXACT_INVERSE_CORE_LABEL_ADMITTED"
            and receipt.get("admitted_metric_ids") == ["CORE_PARENT_NET_PROFIT_YTD"]
        )
    if announcement_id == "1205362872":
        receipt = row.get("net_receivable_reconciliation_receipt") or {}
        return (
            receipt.get("status")
            == "PASS_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED"
            and receipt.get("current_period_identity_passed") is True
            and receipt.get("prior_period_identity_passed") is True
        )
    if announcement_id == "1203181067":
        receipt = row.get("anchored_image_ocr_receipt") or {}
        return (
            receipt.get("status") == "PASS_ANCHORED_IMAGE_OCR_METRICS_ADMITTED"
            and set(receipt.get("admitted_metric_ids") or [])
            == {
                "OPERATING_REVENUE_YTD",
                "OPERATING_PROFIT_YTD",
                "ACCOUNTS_RECEIVABLE_END",
                "INVENTORY_END",
                "TOTAL_LIABILITIES_END",
            }
            and receipt.get("corrected_existing_metric_ids")
            == ["OPERATING_REVENUE_YTD"]
        )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="回放V1.5固定22份PDF及V1.6三类真实缺口PDF"
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
            except Exception as error:  # noqa: BLE001 - 每份失败必须落账
                row = {
                    **target,
                    "document_complete": False,
                    "document_status": "NO_VIEW_V1_6_REPLAY_EXCEPTION",
                    "missing_metrics": list(REQUIRED_METRICS),
                    "metric_count": 0,
                    "metric_values_cny": {},
                    "authoritative_metric_values_match": False,
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            row["targeted_receipt_passed"] = _targeted_receipt_passed(row)
            results.append(row)
            print(
                f"V1.6回放 {completed}/{len(targets)}｜{row['announcement_id']}｜"
                f"{row.get('metric_count', 0)}/9｜逐值一致="
                f"{row.get('authoritative_metric_values_match')}｜"
                f"定向回执={row['targeted_receipt_passed']}",
                flush=True,
            )
    results.sort(key=lambda row: str(row["announcement_id"]))
    legacy = [row for row in results if row["fix_family"] == "V1_5_AUTHORITATIVE_NONREGRESSION"]
    gap = [row for row in results if row["fix_family"] != "V1_5_AUTHORITATIVE_NONREGRESSION"]
    complete_count = sum(bool(row.get("document_complete")) for row in results)
    value_match_count = sum(
        bool(row.get("authoritative_metric_values_match")) for row in results
    )
    targeted_receipt_count = sum(bool(row["targeted_receipt_passed"]) for row in gap)
    passed = (
        len(results) == 25
        and len(legacy) == 22
        and len(gap) == 3
        and complete_count == len(results)
        and value_match_count == len(results)
        and targeted_receipt_count == len(gap)
        and all(row.get("parser_version") == PARSER_VERSION for row in results)
    )
    receipt = {
        "protocol_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_6_REPLAY",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": PARSER_VERSION,
        "target_count": len(results),
        "v1_5_nonregression_target_count": len(legacy),
        "v1_6_gap_target_count": len(gap),
        "complete_count": complete_count,
        "authoritative_metric_value_match_count": value_match_count,
        "targeted_receipt_pass_count": targeted_receipt_count,
        "all_nine_metric_replay_and_authoritative_value_match_passed": passed,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        f"V1.6固定回放完成：完整={complete_count}/{len(results)}；"
        f"逐值一致={value_match_count}/{len(results)}；"
        f"三类定向回执={targeted_receipt_count}/{len(gap)}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
