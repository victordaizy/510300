"""冻结或验证广泛流动 ETF 流动性冲击反转 V1。"""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import load_contract
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v1 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "broad_liquid_etf_liquidity_shock_reversal_v1.yaml"
MANIFEST = ROOT / "config" / "broad_liquid_etf_liquidity_shock_reversal_v1_manifest.json"
TRACKED_FILES = [
    "config/broad_liquid_etf_liquidity_shock_reversal_v1.yaml",
    "docs/BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_V1_SPEC.md",
    "research/broad_liquid_etf_liquidity_shock_reversal_v1.py",
    "scripts/run_broad_liquid_etf_liquidity_shock_reversal_v1.py",
    "scripts/freeze_broad_liquid_etf_liquidity_shock_reversal_v1.py",
]


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """以同目录原子替换方式写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def current_tracked_hashes() -> dict[str, str]:
    """计算全部冻结代码与合同文件的当前哈希。"""

    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少候选冻结文件：{missing}")
    return {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}


def tracked_content_hash(files: dict[str, str]) -> str:
    """计算与路径顺序无关的受控内容聚合哈希。"""

    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_input_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    """冻结全部行情、主表、基准与父协议输入。"""

    snapshot: dict[str, Any] = {}
    for key, relative in contract["inputs"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"缺少候选输入：{relative}")
        snapshot[key] = {
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return snapshot


def freeze() -> dict[str, Any]:
    """在首次读取可见期结果前创建不可变候选清单。"""

    contract = load_contract(CONFIG)
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError(f"父协议清单验证失败：{parent['failures']}")
    files = current_tracked_hashes()
    inputs = current_input_snapshot(contract)
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_V1_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
        "user_transaction_fee_rate_per_leg": float(
            contract["account"]["user_transaction_fee_rate_per_leg"]
        ),
        "minimum_annualized_net_excess": float(
            contract["visible_gates"]["minimum_annualized_net_excess"]
        ),
        "minimum_strategy_net_sharpe": float(
            contract["visible_gates"]["minimum_strategy_net_sharpe"]
        ),
        "visible_partition": [
            contract["historical_partition"]["visible_start"],
            contract["historical_partition"]["visible_end"],
        ],
        "sealed_replication_partition": [
            contract["historical_partition"]["sealed_replication_start"],
            contract["historical_partition"]["sealed_replication_end"],
        ],
        "tracked_file_count": len(files),
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "input_snapshot": inputs,
        "parent_manifest_verification": parent,
        "tests_are_runtime_inputs": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        comparable_fields = (
            "tracked_files",
            "tracked_content_sha256",
            "input_snapshot",
            "visible_partition",
            "sealed_replication_partition",
        )
        if any(existing.get(field) != manifest.get(field) for field in comparable_fields):
            raise RuntimeError("现有冻结清单与当前候选或输入不同；不得覆盖，应创建新版本")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    """验证候选清单、父协议和输入快照均未漂移。"""

    failures: list[str] = []
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
            "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = load_contract(CONFIG)
    live_files = current_tracked_hashes()
    live_inputs = current_input_snapshot(contract)
    parent = verify_parent_manifest()
    if manifest.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if manifest.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if manifest.get("input_snapshot") != live_inputs:
        failures.append("input_snapshot")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_MANIFEST"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": len(live_files),
        "parent_manifest_status": parent["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证广泛流动ETF冲击反转候选")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
