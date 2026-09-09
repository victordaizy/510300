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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from research.csi300_pit_fundamental_underreaction_official_facts_v1_4 import (  # noqa: E402
    PARSER_VERSION,
    REQUIRED_METRICS,
    extract_official_pdf_facts,
)


SOURCE_ROOT = (
    ROOT
    / "tmp/pdfs/csi300_pit_fundamental_underreaction_v1_pilot/"
    "v1_4_replay_sources"
)
SOURCE_MANIFEST = SOURCE_ROOT / "source_manifest.json"
CHECKPOINT_ROOT = (
    ROOT
    / "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
    "checkpoints_v1_3"
)
OUTPUT_ROOT = SOURCE_ROOT / "replay_results"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _checkpoint_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted(CHECKPOINT_ROOT.rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("checkpoint_status") != "PARSED_INCOMPLETE":
            continue
        metadata[str(payload["announcement_id"])] = payload
    return metadata


def _targets(only: set[str]) -> list[dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    metadata = _checkpoint_metadata()
    targets: list[dict[str, Any]] = []
    for row in manifest["results"]:
        announcement_id = str(row["announcement_id"])
        if only and announcement_id not in only:
            continue
        checkpoint = metadata.get(announcement_id)
        if checkpoint is None:
            raise RuntimeError(f"缺少V1.3不完整检查点：{announcement_id}")
        path = ROOT / str(row["path"])
        targets.append(
            {
                "announcement_id": announcement_id,
                "period_type": str(checkpoint["period_type"]),
                "report_period": str(checkpoint["report_period"]),
                "sec_name": str(checkpoint["sec_name"]),
                "expected_sha256": str(checkpoint["official_pdf_sha256"]),
                "path": str(path),
            }
        )
    if not targets:
        raise RuntimeError("固定重放目标为空")
    return targets


def _run_target(target: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(target["path"]))
    content = path.read_bytes()
    observed_sha256 = _sha256(content)
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
        "path": Path(str(target["path"])).relative_to(ROOT).as_posix(),
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
        "v1_4_semantic_table_receipt": result.get("v1_4_semantic_table_receipt") or {},
        "pdfplumber_secondary_text_receipt": result.get("pdfplumber_secondary_text_receipt") or {},
        "ocr_fallback_receipt": result.get("ocr_fallback_receipt") or {},
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
    parser = argparse.ArgumentParser(description="回放V1.3不完整PDF并验证V1.4解析器")
    parser.add_argument("--workers", type=int, default=3)
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
                result = future.result()
            except Exception as error:  # noqa: BLE001 - 每个回放错误需显式落账
                result = {
                    **target,
                    "path": Path(str(target["path"])).relative_to(ROOT).as_posix(),
                    "document_complete": False,
                    "document_status": "NO_VIEW_V1_4_REPLAY_EXCEPTION",
                    "missing_metrics": list(REQUIRED_METRICS),
                    "metric_count": 0,
                    "metric_values_cny": {},
                    "metric_evidence": [],
                    "error": f"{type(error).__name__}: {error}"[:4000],
                }
            results.append(result)
            print(
                f"V1.4回放 {completed}/{len(targets)}｜{result['announcement_id']}｜"
                f"{result['metric_count']}/9｜缺失={','.join(result['missing_metrics']) or '无'}",
                flush=True,
            )
    results.sort(key=lambda item: str(item["announcement_id"]))
    complete_count = sum(bool(item["document_complete"]) for item in results)
    receipt = {
        "protocol_id": "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_OFFICIAL_FACTS_V1_4_REPLAY",
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "parser_version": PARSER_VERSION,
        "target_count": len(results),
        "complete_count": complete_count,
        "incomplete_count": len(results) - complete_count,
        "all_nine_metric_replay_passed": complete_count == len(results),
        "results": results,
        "market_price_read": False,
        "future_return_read": False,
        "return_evaluation": "NOT_ALLOWED",
    }
    _atomic_write_json(RECEIPT_PATH, receipt)
    print(
        f"V1.4固定回放完成：{complete_count}/{len(results)}；回执：{RECEIPT_PATH}",
        flush=True,
    )
    print("本程序未读取市场价格或未来收益。", flush=True)
    return 0 if complete_count == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
