from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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


CHECKPOINT_ROOT = ROOT / (
    "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "checkpoints_v1_6"
)
OUTPUT_ROOT = ROOT / "tmp/pit_v1_7_core_gap_census"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
CORE_ONLY = ("CORE_PARENT_NET_PROFIT_YTD",)
MAXIMUM_PDF_BYTES = 100 * 1024 * 1024
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.cninfo.com.cn/",
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


def _targets() -> list[dict[str, Any]]:
    results = []
    for checkpoint_path in CHECKPOINT_ROOT.rglob("*.json.gz"):
        with gzip.open(checkpoint_path, "rt", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if checkpoint.get("document_terminal") is not True:
            continue
        if checkpoint.get("document_complete") is True:
            continue
        if tuple(checkpoint.get("missing_metrics") or []) != CORE_ONLY:
            continue
        existing_parent = next(
            (
                metric
                for metric in checkpoint.get("metrics") or []
                if metric.get("metric_id") == "PARENT_NET_PROFIT_YTD"
                and metric.get("verification_status")
                == parser.ADMITTED_VERIFICATION_STATUS
            ),
            None,
        )
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
            "existing_parent_net_profit_cny": (
                float(existing_parent["metric_value_cny"])
                if existing_parent is not None
                else None
            ),
            "existing_parent_verification_status": (
                existing_parent.get("verification_status")
                if existing_parent is not None
                else None
            ),
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


def _download(target: dict[str, Any]) -> tuple[Path, str]:
    destination = OUTPUT_ROOT / f"{target['announcement_id']}.pdf"
    route = "REUSED_HASH_VERIFIED_CACHE"
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


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path, acquisition_route = _download(target)
    content = path.read_bytes()
    page_texts, text_receipt = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(text) for text in page_texts
    ]
    result: dict[str, Any] = {
        "metrics": (
            [
                {
                    "metric_id": "PARENT_NET_PROFIT_YTD",
                    "metric_value_cny": target["existing_parent_net_profit_cny"],
                    "verification_status": target[
                        "existing_parent_verification_status"
                    ],
                }
            ]
            if target.get("existing_parent_net_profit_cny") is not None
            else []
        ),
        "missing_metrics": list(parser.REQUIRED_METRICS),
        "document_complete": False,
        "document_status": "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE",
        "parser_version": parser.PARSER_VERSION,
    }
    receipt = parser._add_extended_core_parent_profit(
        result,
        normalized,
        period_type=target["period_type"],
    )
    metric = next(
        (
            row
            for row in result.get("metrics") or []
            if row.get("metric_id") == "CORE_PARENT_NET_PROFIT_YTD"
        ),
        None,
    )
    return {
        **target,
        "path": path.relative_to(ROOT).as_posix(),
        "acquisition_route": acquisition_route,
        "observed_pdf_sha256": _sha256(content),
        "observed_pdf_size_bytes": len(content),
        "pdf_page_count": text_receipt.get("pdf_page_count"),
        "text_character_count": text_receipt.get("text_character_count"),
        "page_error_count": len(text_receipt.get("page_errors") or []),
        "repair_status": receipt.get("status"),
        "repair_method": receipt.get("method"),
        "source_page": receipt.get("source_page"),
        "source_raw_value": receipt.get("source_raw_value"),
        "source_unit": receipt.get("source_unit"),
        "recovered": metric is not None,
        "metric_value_cny": metric.get("metric_value_cny") if metric else None,
        "metric_evidence": metric,
        "targeted_receipt": receipt,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }


def parse_args() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="对V1.6只缺扣非归母净利润的全部官方PDF运行V1.7定向普查"
    )
    argument_parser.add_argument("--workers", type=int, default=4)
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
            except Exception as error:  # 每份失败必须进入收据
                row = {
                    **target,
                    "recovered": False,
                    "repair_status": "NO_VIEW_V1_7_CORE_GAP_CENSUS_EXCEPTION",
                    "error": f"{type(error).__name__}: {error}"[:4000],
                    "market_price_read": False,
                    "future_return_read": False,
                    "return_evaluation": "NOT_ALLOWED",
                }
            results.append(row)
            print(
                f"V1.7扣非归母普查 {completed}/{len(targets)}｜"
                f"{row['announcement_id']}｜恢复={row.get('recovered')}｜"
                f"{row.get('repair_status')}",
                flush=True,
            )
    results.sort(key=lambda row: row["announcement_id"])
    recovered_count = sum(bool(row.get("recovered")) for row in results)
    exception_count = sum("error" in row for row in results)
    receipt = {
        "protocol_id": "CSI300_PIT_OFFICIAL_FACTS_V1_7_CORE_GAP_CENSUS",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": parser.PARSER_VERSION,
        "target_missing_metrics": list(CORE_ONLY),
        "target_count": len(results),
        "recovered_count": recovered_count,
        "still_missing_count": len(results) - recovered_count,
        "exception_count": exception_count,
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        f"V1.7扣非归母普查完成：恢复={recovered_count}/{len(results)}；"
        f"异常={exception_count}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if exception_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
