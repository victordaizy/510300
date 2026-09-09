"""冻结或验证多资产40%净超额与高夏普合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v1_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v1.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V1_SPEC.md",
    "research/multi_asset_annual_excess_40pct_high_sharpe_v1.py",
    "scripts/build_multi_asset_annual_excess_40pct_high_sharpe_v1.py",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v1.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    return {relative: sha256(ROOT / relative) for relative in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    payload = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze() -> dict:
    files = current_hashes()
    payload = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V1_MANIFEST",
        "status": "FROZEN_BEFORE_CANDIDATE_SCREEN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "tracked_file_count": len(files),
        "tracked_files": files,
        "content_sha256": content_hash(files),
        "tests_are_runtime_inputs": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("tracked_files") != files:
            raise RuntimeError("冻结清单已存在且内容不同，禁止覆盖")
        return existing
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def verify() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError("多资产40%高夏普冻结清单尚未创建")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live = current_hashes()
    failures: list[str] = []
    if manifest.get("status") != "FROZEN_BEFORE_CANDIDATE_SCREEN":
        failures.append("status")
    if float(manifest.get("initial_capital_cny", -1.0)) != 500000.0:
        failures.append("initial_capital_cny")
    if float(manifest.get("user_transaction_fee_rate_per_leg", -1.0)) != 0.0001:
        failures.append("user_transaction_fee_rate_per_leg")
    if float(manifest.get("minimum_annualized_net_excess", -1.0)) != 0.40:
        failures.append("minimum_annualized_net_excess")
    if float(manifest.get("minimum_strategy_net_sharpe", -1.0)) != 1.50:
        failures.append("minimum_strategy_net_sharpe")
    if manifest.get("tracked_files") != live:
        failures.append("tracked_files")
    if manifest.get("content_sha256") != content_hash(live):
        failures.append("content_sha256")
    return {
        "status": "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_MANIFEST_VERIFIED"
        if not failures
        else "FAILED_MULTI_ASSET_40PCT_HIGH_SHARPE_MANIFEST",
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(MANIFEST),
        "content_sha256": content_hash(live),
        "tracked_file_count": len(live),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证多资产40%高夏普合同")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.mode == "verify" and result["failure_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
