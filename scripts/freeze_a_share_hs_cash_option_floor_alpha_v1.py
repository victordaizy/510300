from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_cash_option_floor_alpha_v1 import (  # noqa: E402
    EXPECTED_EVENT_COUNT,
    PROTOCOL_ID,
    validate_protocol_config,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
CONFIG_PATH = ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1.json"
PROTOCOL_STATUS = "FROZEN_CASH_OPTION_FLOOR_ALPHA_BEFORE_FIRST_MARKET_PRICE_VALUE_READ"
CORE_FILES = [
    ROOT / "config/a_share_hs_official_cash_option_floor_alpha_v1.json",
    ROOT / "docs/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_PROTOCOL.md",
    ROOT / "research/a_share_hs_cash_option_floor_alpha_v1.py",
    ROOT / "scripts/freeze_a_share_hs_cash_option_floor_alpha_v1.py",
    ROOT / "tests/test_a_share_hs_cash_option_floor_alpha_v1.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_protocol_config(config)
    return config


def artifact_path(config: dict[str, Any], key: str) -> Path:
    return ROOT / str(config["artifacts"][key])


def verify_upstream(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream"]
    manifest_path = ROOT / str(upstream["terms_archive_manifest"])
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != str(upstream["terms_archive_manifest_sha256"]):
        raise RuntimeError("现金选择权条款冻结清单 SHA-256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != upstream["terms_archive_required_status"]:
        raise RuntimeError(f"现金选择权条款冻结状态不匹配：{manifest.get('status')}")
    failures: list[dict[str, Any]] = []
    for item_path, expected_sha in (manifest.get("files") or {}).items():
        path = ROOT / str(item_path)
        actual_sha = sha256_file(path) if path.exists() else None
        if actual_sha != expected_sha:
            failures.append(
                {
                    "path": str(item_path),
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
    if failures:
        raise RuntimeError(
            "现金选择权条款冻结文件漂移："
            + json.dumps(failures, ensure_ascii=False, default=str)
        )

    outcomes_path = ROOT / str(upstream["terms_outcomes"])
    outcomes_sha = sha256_file(outcomes_path)
    if outcomes_sha != upstream["terms_outcomes_sha256"]:
        raise RuntimeError("现金选择权条款结果 SHA-256 不匹配")
    outcomes = pd.read_parquet(outcomes_path)
    if len(outcomes) != EXPECTED_EVENT_COUNT:
        raise RuntimeError(f"现金选择权条款事件数不是 {EXPECTED_EVENT_COUNT}")
    if outcomes["event_id"].astype(str).duplicated().any():
        raise RuntimeError("现金选择权条款事件ID重复")
    if not outcomes["classification_status"].astype(str).eq(
        upstream["required_terms_status"]
    ).all():
        raise RuntimeError("存在非完整现金选择权条款事件")
    if not outcomes["terms_complete"].fillna(False).astype(bool).all():
        raise RuntimeError("存在 terms_complete=false 的现金选择权条款事件")
    for field in (
        "market_price_read",
        "future_return_read",
        "observed_premium_filter_applied",
        "liquidity_filter_applied",
        "cash_option_price_filter_applied",
    ):
        if outcomes[field].fillna(True).astype(bool).any():
            raise RuntimeError(f"条款上游治理字段不是全 false：{field}")
    if not outcomes["cash_option_price"].gt(0).all():
        raise RuntimeError("现金选择权条款存在非正价格")
    if not outcomes["currency"].astype(str).eq("CNY").all():
        raise RuntimeError("现金选择权条款币种不是全人民币")
    if not outcomes["share_scope"].astype(str).eq("A_SHARE").all():
        raise RuntimeError("现金选择权条款范围不是全 A 股")
    return {
        "status": "PASS_FROZEN_CASH_OPTION_TERMS_VERIFIED_BEFORE_MARKET_READ",
        "event_count": len(outcomes),
        "distinct_event_count": int(outcomes["event_id"].astype(str).nunique()),
        "terms_archive_manifest": relative(manifest_path),
        "terms_archive_manifest_sha256": actual_manifest_sha,
        "terms_outcomes": relative(outcomes_path),
        "terms_outcomes_sha256": outcomes_sha,
        "market_price_value_read": False,
        "future_subject_return_read": False,
    }


def verify_reference_inputs(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for item in config["market_data_contract"]["preexisting_reference_inputs"]:
        path = ROOT / str(item["path"])
        if not path.exists():
            raise RuntimeError(f"冻结参考输入不存在：{item['path']}")
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"]:
            raise RuntimeError(f"冻结参考输入 SHA-256 漂移：{item['path']}")
        parquet = pq.ParquetFile(path)
        checks.append(
            {
                "path": str(item["path"]),
                "sha256": actual_sha,
                "role": str(item["role"]),
                "row_count_from_parquet_metadata": int(parquet.metadata.num_rows),
                "row_group_count_from_parquet_metadata": int(parquet.metadata.num_row_groups),
                "schema_fields": list(parquet.schema_arrow.names),
                "price_value_read": False,
            }
        )
    return {
        "status": "PASS_PREEXISTING_REFERENCE_INPUT_HASH_AND_SCHEMA_METADATA_VERIFIED",
        "file_count": len(checks),
        "files": checks,
        "price_value_read": False,
    }


def validate_plan(config: dict[str, Any]) -> dict[str, Any]:
    upstream = verify_upstream(config)
    references = verify_reference_inputs(config)
    missing_core_files = [relative(path) for path in CORE_FILES if not path.exists()]
    return {
        "status": (
            "PASS_CASH_OPTION_FLOOR_ALPHA_PROTOCOL_PLAN_VALIDATED"
            if not missing_core_files
            else "FAILED_CASH_OPTION_FLOOR_ALPHA_PROTOCOL_CORE_FILES_MISSING"
        ),
        "upstream_verification": upstream,
        "reference_input_verification": references,
        "missing_core_files": missing_core_files,
        "protocol_manifest_exists": artifact_path(config, "protocol_manifest").exists(),
        "market_schema_and_coverage_metadata_inspected": True,
        "market_price_value_read": False,
        "future_subject_return_read": False,
    }


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    upstream = verify_upstream(config)
    references = verify_reference_inputs(config)
    path = artifact_path(config, "protocol_manifest")
    if path.exists():
        raise RuntimeError("现金选择权底价协议冻结清单已存在，禁止覆盖")
    missing = [relative(item) for item in CORE_FILES if not item.exists()]
    if missing:
        raise RuntimeError(f"现金选择权底价协议核心文件缺失：{missing}")
    files = [{"path": relative(item), "sha256": sha256_file(item)} for item in CORE_FILES]
    digest_payload = {item["path"]: item["sha256"] for item in files}
    manifest = {
        "protocol_id": f"{PROTOCOL_ID}_PROTOCOL_FREEZE",
        "status": PROTOCOL_STATUS,
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "evidence_cutoff": config["evidence_cutoff"],
        "files": files,
        "content_sha256": hashlib.sha256(
            canonical_json(digest_payload).encode("utf-8")
        ).hexdigest(),
        "upstream_verification": upstream,
        "reference_input_verification": references,
        "frozen_event_count": EXPECTED_EVENT_COUNT,
        "minimum_net_conditional_floor_return": config["screening"][
            "minimum_net_conditional_floor_return"
        ],
        "board_lot_shares": config["clock_and_entry"]["board_lot_shares"],
        "minimum_price_qualified_events": config["acceptance_gates"][
            "minimum_price_qualified_events"
        ],
        "market_schema_and_coverage_metadata_inspected": True,
        "market_price_value_read": False,
        "future_subject_return_read": False,
        "price_screen_completed": False,
        "parameter_search_performed": False,
        "liquidity_filter_applied": False,
        "position_mapping": False,
        "order_generation": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    upstream = verify_upstream(config)
    references = verify_reference_inputs(config)
    path = artifact_path(config, "protocol_manifest")
    if not path.exists():
        raise RuntimeError("现金选择权底价协议冻结清单不存在")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    if manifest.get("status") != PROTOCOL_STATUS:
        failures.append(
            {"failure": "STATUS", "expected": PROTOCOL_STATUS, "actual": manifest.get("status")}
        )
    if manifest.get("frozen_event_count") != EXPECTED_EVENT_COUNT:
        failures.append(
            {
                "failure": "EVENT_COUNT",
                "expected": EXPECTED_EVENT_COUNT,
                "actual": manifest.get("frozen_event_count"),
            }
        )
    for flag in (
        "market_price_value_read",
        "future_subject_return_read",
        "price_screen_completed",
        "parameter_search_performed",
        "liquidity_filter_applied",
        "position_mapping",
        "order_generation",
        "shadow_authorized",
        "live_authorized",
    ):
        if manifest.get(flag) is not False:
            failures.append({"failure": "GOVERNANCE_FLAG", "field": flag, "actual": manifest.get(flag)})
    for item in manifest.get("files") or []:
        item_path = ROOT / str(item["path"])
        actual_sha = sha256_file(item_path) if item_path.exists() else None
        if actual_sha != item["sha256"]:
            failures.append(
                {
                    "failure": "FILE_HASH",
                    "path": item["path"],
                    "expected": item["sha256"],
                    "actual": actual_sha,
                }
            )
    if failures:
        raise RuntimeError(json.dumps(failures, ensure_ascii=False, default=str))
    return {
        "status": "PASS_CASH_OPTION_FLOOR_ALPHA_FROZEN_PROTOCOL_VERIFIED",
        "protocol_manifest": relative(path),
        "protocol_manifest_sha256": sha256_file(path),
        "content_sha256": manifest["content_sha256"],
        "checked_core_file_count": len(manifest["files"]),
        "upstream_verification": upstream,
        "reference_input_verification": references,
        "market_price_value_read": False,
        "future_subject_return_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在首次市场价格值读取前冻结 A 股现金选择权高赔率底价 Alpha 协议"
    )
    parser.add_argument(
        "--mode",
        choices=("validate-plan", "freeze-protocol", "verify-protocol"),
        default="validate-plan",
    )
    args = parser.parse_args()
    config = load_config()
    if args.mode == "validate-plan":
        result = validate_plan(config)
    elif args.mode == "freeze-protocol":
        result = freeze_protocol(config)
    else:
        result = verify_protocol(config)
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
