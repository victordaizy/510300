"""在桶3与桶4未来收益下载前冻结分层低波动公式和评估器。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml"
FROZEN_FILES = (
    "config/a_share_bucket34_time_holdout_stratified_lowvol_v1.yaml",
    "docs/A_SHARE_BUCKET34_TIME_HOLDOUT_STRATIFIED_LOWVOL_V1_SPEC.md",
    "research/a_share_hash_holdout_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_a_share_bucket34_time_holdout_stratified_lowvol_v1.py",
    "scripts/download_a_share_bucket34_time_holdout_stratified_lowvol_v1.py",
    "scripts/run_a_share_bucket34_time_holdout_stratified_lowvol_v1.py",
    "tests/test_a_share_hash_holdout_alpha_v1.py",
    "tests/test_small_account_cross_sectional.py",
    "tests/test_a_share_bucket34_time_holdout_stratified_lowvol_v1.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def verify_protocol() -> tuple[dict, dict]:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "FORMULA_FROZEN_BEFORE_BUCKET34_FUTURE_RETURN_DOWNLOAD":
        raise RuntimeError("桶3+4未来盲测协议冻结状态错误")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    if sha256(ROOT / contract["inputs"]["master"]) != manifest["master_sha256"]:
        changed.append(contract["inputs"]["master"])
    if changed:
        raise RuntimeError(f"桶3+4协议冻结后变化：{sorted(set(changed))}")
    return contract, manifest


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths = contract["paths"]
    manifest_path = ROOT / paths["protocol_manifest"]
    if manifest_path.exists():
        raise FileExistsError("桶3+4协议清单已存在，禁止覆盖")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    forbidden = [
        paths[key]
        for key in (
            "checkpoint_directory", "holdout_panel", "benchmark", "data_status", "features",
            "targets", "base_ledger", "base_trades", "stress_ledger", "stress_trades", "result_json",
        )
    ]
    existing = [name for name in forbidden if (ROOT / name).exists()]
    if existing:
        raise RuntimeError(f"公式冻结前已有桶3+4未来收益或结果：{existing}")
    if contract["split"]["future_holdout_buckets"] != [3, 4]:
        raise ValueError("未来留出桶必须精确为3和4")
    if contract["formula"]["factor_count"] != 1 or contract["formula"]["fit_required"]:
        raise ValueError("冻结公式必须是无需拟合的单因子低波动公式")
    if contract["selection"]["holdings"] != 2 or contract["selection"]["retention_score_rank_within_bucket"] != 1:
        raise ValueError("冻结选择规则必须是两桶各第一名")
    tests = (
        "tests/test_a_share_hash_holdout_alpha_v1.py",
        "tests/test_small_account_cross_sectional.py",
        "tests/test_a_share_bucket34_time_holdout_stratified_lowvol_v1.py",
    )
    test = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "state": "FORMULA_FROZEN_BEFORE_BUCKET34_FUTURE_RETURN_DOWNLOAD",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "project_id": contract["protocol"]["project_id"],
        "future_holdout_buckets": [3, 4],
        "future_bucket3_and_bucket4_returns_downloaded_before_freeze": False,
        "factor_count": 1,
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "master_sha256": sha256(ROOT / contract["inputs"]["master"]),
        "test_result": test.stdout.strip(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "未来桶": [3, 4], "未来收益已下载": False, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
