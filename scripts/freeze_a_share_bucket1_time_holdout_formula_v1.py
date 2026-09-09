"""在桶1未来收益下载前冻结四因子公式与评估器。"""

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
CONFIG_FILE = ROOT / "config" / "a_share_bucket1_time_holdout_formula_v1.yaml"
FROZEN_FILES = (
    "config/a_share_bucket1_time_holdout_formula_v1.yaml",
    "docs/A_SHARE_BUCKET1_TIME_HOLDOUT_FORMULA_V1_SPEC.md",
    "research/a_share_hash_holdout_alpha_v1.py",
    "research/small_account_cross_sectional.py",
    "scripts/freeze_a_share_bucket1_time_holdout_formula_v1.py",
    "scripts/download_a_share_bucket1_time_holdout_formula_v1.py",
    "scripts/run_a_share_bucket1_time_holdout_formula_v1.py",
    "tests/test_a_share_hash_holdout_alpha_v1.py",
    "tests/test_small_account_cross_sectional.py",
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
    if manifest.get("state") != "FORMULA_FROZEN_BEFORE_BUCKET1_FUTURE_RETURN_DOWNLOAD":
        raise RuntimeError("桶1未来盲测协议冻结状态错误")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    if sha256(ROOT / contract["inputs"]["master"]) != manifest["master_sha256"]:
        changed.append(contract["inputs"]["master"])
    if changed:
        raise RuntimeError(f"桶1协议冻结后变化：{sorted(set(changed))}")
    return contract, manifest


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    if manifest_path.exists():
        raise FileExistsError("桶1协议清单已存在，禁止覆盖")
    missing = [name for name in FROZEN_FILES if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    paths = contract["paths"]
    forbidden = [paths[key] for key in ("checkpoint_directory", "holdout_panel", "benchmark", "data_status", "features", "targets", "result_json")]
    existing = [name for name in forbidden if (ROOT / name).exists()]
    if existing:
        raise RuntimeError(f"公式冻结前已有桶1未来收益或结果：{existing}")
    if contract["formula"]["factor_count"] > 10 or contract["formula"]["fit_required"]:
        raise ValueError("公式因子数量或拟合约束错误")
    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_a_share_hash_holdout_alpha_v1.py", "tests/test_small_account_cross_sectional.py", "-q"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if test.returncode:
        raise RuntimeError(test.stdout + "\n" + test.stderr)
    manifest = {
        "state": "FORMULA_FROZEN_BEFORE_BUCKET1_FUTURE_RETURN_DOWNLOAD",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "project_id": contract["protocol"]["project_id"],
        "future_holdout_bucket": 1,
        "future_bucket1_returns_downloaded_before_freeze": False,
        "factor_count": contract["formula"]["factor_count"],
        "frozen_files": {name: sha256(ROOT / name) for name in FROZEN_FILES},
        "master_sha256": sha256(ROOT / contract["inputs"]["master"]),
        "test_result": test.stdout.strip(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "因子": manifest["factor_count"], "桶1未来收益已下载": False, "测试": manifest["test_result"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
